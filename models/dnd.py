from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(slots=True) 
class MemoryEntry: 
    """
    Represents one entry stored in the DND.

    # ! WARNING: NOT USED ANYWHERE. Can be removed

    Attributes:
        key:
            State representation.

        value:
            Stored value associated with the representation.

        auxiliary:
            Optional data associated with the key, such as latent
            variances or other metadata required by custom similarity
            functions.
    """

    key: torch.Tensor
    value: torch.Tensor
    auxiliary: torch.Tensor | None = None


@dataclass(slots=True)
class LookupResult:
    value: torch.Tensor
    neighbor_indices: torch.Tensor | None = None
    neighbor_generations: torch.Tensor | None = None
    similarities: torch.Tensor | None = None
    neighbor_keys: torch.Tensor | None = None
    neighbor_values: torch.Tensor | None = None
    neighbor_auxiliary: torch.Tensor | None = None


class DND:
    """
    Differentiable Neural Dictionary.

    Responsible for storing memory entries and performing nearest
    neighbor lookup. Learning rules are implemented separately by
    update strategies.
    """

    def __init__(
        self,
        representation_dim: int,
        value_dim: int = 1,
        use_auxiliary: bool = False,
        auxiliary_dim: int = 1, 
        max_memory: int = 500000,
        num_neighbors: int = 50,
        neighbor_index = None,
        similarity_function=None,
        learning_rate: float = 1e-3,
        device = torch.device("cpu"),
    ):
        self.representation_dim = representation_dim
        self.value_dim = value_dim
        self.use_auxiliary = use_auxiliary
        self.auxiliary_dim = auxiliary_dim

        self.max_memory = max_memory
        self.num_neighbors = num_neighbors
        
        self.neighbor_index = neighbor_index
        self.similarity_function = similarity_function
        self.learning_rate = learning_rate
        self.device = device

        self.keys = torch.empty(
            (self.max_memory, self.representation_dim),
            dtype=torch.float,
            device=self.device,
        )
        self.values = torch.empty(
            (self.max_memory, self.value_dim),
            dtype=torch.float,
            device=self.device,
        )
        self.generations = torch.zeros(
            self.max_memory,
            dtype=torch.long,
            device=self.device,
        )

        if self.use_auxiliary:
            self.auxiliary = torch.empty(
                (self.max_memory, auxiliary_dim),
                dtype=torch.float,
                device=self.device,
            )

        self.write_index = 0
        self.memory_size = 0

        self._pending_keys = []
        self._pending_values = []
        self._pending_auxiliary = [] if self.use_auxiliary else None

        self._stale_index = True

        self.key_optimizer: torch.optim.Optimizer | None = None
        self.optimizer_stale = True 
        # This is becomes True when new keys inserted, which is in the 'commit()'.
        # If it is False, unnecessary optimizer initializations are avoided during key update.

    def __len__(self) -> int:
        """
        Returns the number of committed memory entries.
        """
        return self.memory_size
    
    def get_index(self, key: torch.Tensor) -> int:
        """
        Returns the index of the given key in the searchable memory.s
        """
        return self.neighbor_index.find(key)

    def get_value(self, index: int) -> torch.Tensor:
        return self.values[index]

    def clear(self):
        """
        Removes all committed and pending memory entries.
        """

        self.keys = None
        self._pending_keys.clear()

        self.values = None
        self._pending_values.clear()
        
        if self.use_auxiliary:
            self.auxiliary = None
            self._pending_auxiliary.clear()
        
        self._stale_index = True

    def insert(self, 
               key: list[torch.Tensor], 
               value: list[torch.Tensor], 
               auxiliary = None):
        """
        Stages a memory entry for insertion.

        The entry is not added to the searchable memory until
        ``commit()`` is called.
        """

        self._pending_keys.extend(key)
        self._pending_values.extend(value)
        
        if self.use_auxiliary:
            self._pending_auxiliary.extend(auxiliary)

        self._stale_index = True

    def commit(self):
        """
        Commits all pending memory entries.

        Pending entries become part of the searchable memory. When the
        memory capacity is reached, the oldest entries are overwritten in a
        circular manner.
        """

        if len(self._pending_keys) == 0:
            return

        pending_size = len(self._pending_keys)

        pending_keys = torch.stack(self._pending_keys).squeeze(dim=1)
        pending_values = torch.stack(self._pending_values)

        pending_auxiliary = None
        if self.use_auxiliary:
            pending_auxiliary = torch.stack(self._pending_auxiliary)

        if self.memory_size < self.max_memory:

            free = self.max_memory - self.memory_size
            append_count = min(free, pending_size)

            append_indices = torch.arange(
                self.memory_size,
                self.memory_size + append_count,
                device=self.keys.device,
            )


            print("values shape", self.values.shape)
            print("pending_values", pending_values.shape)
            print("values[append_indices]", self.values[append_indices].shape)
            print("pending_values[:append_count]", pending_values[:append_count].shape)

            self.generations[append_indices] += 1

            self.keys[append_indices] = pending_keys[:append_count]
            self.values[append_indices] = pending_values[:append_count]

            if self.use_auxiliary:
                self.auxiliary[append_indices] = pending_auxiliary[:append_count]

            self.memory_size += append_count

            pending_keys = pending_keys[append_count:]
            pending_values = pending_values[append_count:]

            if self.use_auxiliary:
                pending_auxiliary = pending_auxiliary[append_count:]

            pending_size -= append_count

        if pending_size > 0:

            write_indices = (
                torch.arange(
                    self.write_index,
                    self.write_index + pending_size,
                    device=self.keys.device,
                )
                % self.max_memory
            )

            self.generations[write_indices] += 1

            self.keys[write_indices] = pending_keys
            self.values[write_indices] = pending_values

            if self.use_auxiliary:
                self.auxiliary[write_indices] = pending_auxiliary

            self.write_index = (self.write_index + pending_size) % self.max_memory

        self._pending_keys.clear()
        self._pending_values.clear()

        if self.use_auxiliary:
            self._pending_auxiliary.clear()

        self._stale_index = True
        self.optimizer_stale = True
        
    def contains(self, key: torch.Tensor,) -> bool:
        """
        Returns whether the given key already exists in committed memory.
        
        !NOTE: If this functions cause any bottleneck later, its functionality will be replaced with hash-map.
        """

        if self.keys is None:
            return False

        key = key.reshape(1, -1)

        return torch.any(
            torch.all(self.keys == key, dim=1)
            ).item()
    
    def build_index(self):
        """
        Rebuilds the neighbor index.
        """
        self.neighbor_index.build(self.keys)
    
    def lookup(
        self,
        key: torch.Tensor,
        return_indices: bool = False,
        return_similarities: bool = False,
        return_neighbors: bool = False,
        track_key_updates: bool = False # This one not required at the moment. It might be used in the future for optimization.
    ) -> LookupResult:
        """
        Retrieves the nearest neighbors of the given key and estimates its
        value using the configured similarity function.

        Additional neighbor information is computed and returned only when
        requested.
        """

        if len(self) == 0:
            raise RuntimeError("Cannot perform lookup on an empty DND.")

        if self._stale_index:
            self.build_index()
            self._stale_index = False

        neighbor_indices = self.neighbor_index.search(key, self.num_neighbors)
        
        neighbor_generations = self.generations[neighbor_indices]
        neighbor_keys = self.keys[neighbor_indices]
        neighbor_values = self.values[neighbor_indices]

        neighbor_auxiliary = None

        if self.use_auxiliary:
            neighbor_auxiliary = self.auxiliary[neighbor_indices]

        similarities = self.similarity_function(
            key=key,
            neighbor_keys=neighbor_keys,
            neighbor_auxiliary=neighbor_auxiliary,
        )

        estimated_value = (
            (similarities.unsqueeze(-1) * neighbor_values).sum(dim=0)
            / similarities.sum()
        )

        result = LookupResult(value=estimated_value)

        if return_indices:
            result.neighbor_indices = neighbor_indices
            result.neighbor_generations = neighbor_generations

        if return_similarities:
            result.similarities = similarities

        if return_neighbors:
            result.neighbor_keys = neighbor_keys
            result.neighbor_values = neighbor_values

            if self.use_auxiliary:
                result.neighbor_auxiliary = neighbor_auxiliary

        return result

    def update(
        self,
        indices: torch.Tensor,
        values: torch.Tensor | None = None,
        changes: torch.Tensor | None = None,
        keys: torch.Tensor | None = None,
        auxiliary: torch.Tensor | None = None,
        ):
        """
        Updates existing memory entries.

        Any of ``values``, ``changes``, ``keys`` or ``auxiliary`` may be omitted.
        Only the provided memory components are updated.
        """

        if len(self) == 0:
            raise RuntimeError("Cannot update an empty DND.")

        if (values is None and keys is None and auxiliary is None):
            raise ValueError(
                "At least one of values, keys or auxiliary must be provided."
            )

        if keys is not None:
            self.keys[indices] = keys
            self._stale_index = True

        if values is not None:
            self.values[indices] = values

        if changes is not None:
            self.values[indices] += changes

        if auxiliary is not None:

            if not self.use_auxiliary:
                raise RuntimeError(
                    "This DND was created without auxiliary memory."
                )

            self.auxiliary[indices] = auxiliary
            
    def initialize_key_optimizer(self) -> None:
        """
        Creates the key optimizer if it does not exist or if the key tensor
        has been structurally modified.
        """

        if not self.trainable_keys:
            return

        if (self.key_optimizer is None or self.optimizer_stale):
            
            self.key_optimizer = torch.optim.RMSprop([self.keys], lr=self.learning_rate)
            self.optimizer_stale = False

    def state_dict(self) -> dict:
        """
        Returns the state dictionary of the DND.
        """
        return {
            "keys": self.keys,
            "values": self.values,
            "auxiliary": self.auxiliary,
        }
    