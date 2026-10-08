use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::ops::{Deref, DerefMut};
use thiserror::Error;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// Standard physical page/sector size for NVMe SSD direct I/O (4 KB).
pub const PAGE_SIZE: usize = 4096;

/// Default maximum out-degree (neighbor count) for the Vamana graph index.
pub const MAX_NEIGHBORS: usize = 64;

/// Maximum vector dimension that fits in a single 4 KB sector alongside metadata.
pub const MAX_DIM_SINGLE_PAGE: usize = 768;

/// Maximum vector dimension for a dual-page (8 KB) sector block (e.g. OpenAI embeddings).
pub const MAX_DIM_1536: usize = 1536;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum StorageError {
    #[error("Buffer length too small: expected at least {expected} bytes, got {actual}")]
    BufferTooSmall { expected: usize, actual: usize },

    #[error("Buffer address 0x{address:x} is misaligned: must be a multiple of {required_alignment}")]
    MisalignedBuffer {
        address: usize,
        required_alignment: usize,
    },

    #[error("Vector dimension {dim} exceeds maximum supported capacity {max_capacity}")]
    VectorDimensionExceeded { dim: usize, max_capacity: usize },

    #[error("Neighbor count {count} exceeds maximum supported capacity {max_capacity}")]
    NeighborsExceeded { count: usize, max_capacity: usize },

    #[error("Memory layout or allocation error: {0}")]
    AllocationError(String),

    #[error("Zero-copy cast error: {0}")]
    CastError(String),

    #[error("I/O error: {0}")]
    IoError(String),
}

/// A zero-copy, 4 KB page-aligned disk sector representation of a vector graph node.
///
/// Strictly satisfies:
/// - `std::mem::align_of::<DiskNode>() == 4096`
/// - `std::mem::size_of::<DiskNode>() == 4096`
#[repr(C, align(4096))]
#[derive(Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct DiskNode {
    /// Node identifier in the graph index.
    pub node_id: u32,
    /// Number of active floating-point dimensions.
    pub vector_dim: u32,
    /// Number of active neighbor connections in the adjacency list.
    pub num_neighbors: u32,
    /// Status flags or reserved header field.
    pub flags: u32,
    /// Outgoing neighbor node IDs.
    pub neighbors: [u32; MAX_NEIGHBORS],
    /// Uncompressed float vector data.
    pub vector_data: [f32; MAX_DIM_SINGLE_PAGE],
    /// Explicit padding to ensure struct size is exactly 4,096 bytes with zero bleed.
    pub _padding: [u8; 752],
}

impl Default for DiskNode {
    fn default() -> Self {
        Self {
            node_id: 0,
            vector_dim: 0,
            num_neighbors: 0,
            flags: 0,
            neighbors: [0u32; MAX_NEIGHBORS],
            vector_data: [0.0f32; MAX_DIM_SINGLE_PAGE],
            _padding: [0u8; 752],
        }
    }
}

impl DiskNode {
    /// Constructs a new DiskNode initialized with vector and neighbor data.
    pub fn new(node_id: u32, vector: &[f32], neighbors: &[u32]) -> Result<Self, StorageError> {
        let mut node = Self::default();
        node.node_id = node_id;
        node.set_vector(vector)?;
        node.set_neighbors(neighbors)?;
        Ok(node)
    }

    /// Returns the active uncompressed float vector slice.
    #[inline]
    pub fn vector(&self) -> &[f32] {
        let dim = self.vector_dim as usize;
        debug_assert!(dim <= MAX_DIM_SINGLE_PAGE);
        &self.vector_data[..dim]
    }

    /// Sets the node's vector elements.
    pub fn set_vector(&mut self, vec: &[f32]) -> Result<(), StorageError> {
        if vec.len() > MAX_DIM_SINGLE_PAGE {
            return Err(StorageError::VectorDimensionExceeded {
                dim: vec.len(),
                max_capacity: MAX_DIM_SINGLE_PAGE,
            });
        }
        self.vector_dim = vec.len() as u32;
        self.vector_data[..vec.len()].copy_from_slice(vec);
        Ok(())
    }

    /// Returns the active neighbor node IDs in the adjacency list.
    #[inline]
    pub fn neighbors(&self) -> &[u32] {
        let count = self.num_neighbors as usize;
        debug_assert!(count <= MAX_NEIGHBORS);
        &self.neighbors[..count]
    }

    /// Sets the node's neighbor list.
    pub fn set_neighbors(&mut self, neighbors: &[u32]) -> Result<(), StorageError> {
        if neighbors.len() > MAX_NEIGHBORS {
            return Err(StorageError::NeighborsExceeded {
                count: neighbors.len(),
                max_capacity: MAX_NEIGHBORS,
            });
        }
        self.num_neighbors = neighbors.len() as u32;
        self.neighbors[..neighbors.len()].copy_from_slice(neighbors);
        Ok(())
    }

    /// Casts this node to raw bytes without copying.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        IntoBytes::as_bytes(self)
    }

    /// Casts this node to mutable raw bytes without copying.
    #[inline]
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        IntoBytes::as_mut_bytes(self)
    }

    /// Casts a 4 KB page-aligned byte slice into a `&DiskNode` with zero heap allocations.
    #[inline]
    pub fn from_bytes(bytes: &[u8]) -> Result<&Self, StorageError> {
        if bytes.len() < PAGE_SIZE {
            return Err(StorageError::BufferTooSmall {
                expected: PAGE_SIZE,
                actual: bytes.len(),
            });
        }

        let addr = bytes.as_ptr() as usize;
        if addr % PAGE_SIZE != 0 {
            return Err(StorageError::MisalignedBuffer {
                address: addr,
                required_alignment: PAGE_SIZE,
            });
        }

        FromBytes::ref_from_prefix(bytes)
            .map(|(node, _)| node)
            .map_err(|e| StorageError::CastError(format!("{:?}", e)))
    }

    /// Casts a 4 KB page-aligned mutable byte slice into a `&mut DiskNode` with zero heap allocations.
    #[inline]
    pub fn from_bytes_mut(bytes: &mut [u8]) -> Result<&mut Self, StorageError> {
        if bytes.len() < PAGE_SIZE {
            return Err(StorageError::BufferTooSmall {
                expected: PAGE_SIZE,
                actual: bytes.len(),
            });
        }

        let addr = bytes.as_ptr() as usize;
        if addr % PAGE_SIZE != 0 {
            return Err(StorageError::MisalignedBuffer {
                address: addr,
                required_alignment: PAGE_SIZE,
            });
        }

        FromBytes::mut_from_prefix(bytes)
            .map(|(node, _)| node)
            .map_err(|e| StorageError::CastError(format!("{:?}", e)))
    }
}

/// A zero-copy, 8 KB (dual-page) sector block supporting 1536-dimensional embeddings.
///
/// Strictly satisfies:
/// - `std::mem::align_of::<DiskNode1536>() == 4096`
/// - `std::mem::size_of::<DiskNode1536>() == 8192` (`size_of % 4096 == 0`)
#[repr(C, align(4096))]
#[derive(Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct DiskNode1536 {
    pub node_id: u32,
    pub vector_dim: u32,
    pub num_neighbors: u32,
    pub flags: u32,
    pub neighbors: [u32; MAX_NEIGHBORS],
    pub vector_data: [f32; MAX_DIM_1536],
    pub _padding: [u8; 1776],
}

impl Default for DiskNode1536 {
    fn default() -> Self {
        Self {
            node_id: 0,
            vector_dim: 0,
            num_neighbors: 0,
            flags: 0,
            neighbors: [0u32; MAX_NEIGHBORS],
            vector_data: [0.0f32; MAX_DIM_1536],
            _padding: [0u8; 1776],
        }
    }
}

impl DiskNode1536 {
    pub fn new(node_id: u32, vector: &[f32], neighbors: &[u32]) -> Result<Self, StorageError> {
        let mut node = Self::default();
        node.node_id = node_id;
        node.set_vector(vector)?;
        node.set_neighbors(neighbors)?;
        Ok(node)
    }

    #[inline]
    pub fn vector(&self) -> &[f32] {
        let dim = self.vector_dim as usize;
        debug_assert!(dim <= MAX_DIM_1536);
        &self.vector_data[..dim]
    }

    pub fn set_vector(&mut self, vec: &[f32]) -> Result<(), StorageError> {
        if vec.len() > MAX_DIM_1536 {
            return Err(StorageError::VectorDimensionExceeded {
                dim: vec.len(),
                max_capacity: MAX_DIM_1536,
            });
        }
        self.vector_dim = vec.len() as u32;
        self.vector_data[..vec.len()].copy_from_slice(vec);
        Ok(())
    }

    #[inline]
    pub fn neighbors(&self) -> &[u32] {
        let count = self.num_neighbors as usize;
        debug_assert!(count <= MAX_NEIGHBORS);
        &self.neighbors[..count]
    }

    pub fn set_neighbors(&mut self, neighbors: &[u32]) -> Result<(), StorageError> {
        if neighbors.len() > MAX_NEIGHBORS {
            return Err(StorageError::NeighborsExceeded {
                count: neighbors.len(),
                max_capacity: MAX_NEIGHBORS,
            });
        }
        self.num_neighbors = neighbors.len() as u32;
        self.neighbors[..neighbors.len()].copy_from_slice(neighbors);
        Ok(())
    }

    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        IntoBytes::as_bytes(self)
    }

    #[inline]
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        IntoBytes::as_mut_bytes(self)
    }

    #[inline]
    pub fn from_bytes(bytes: &[u8]) -> Result<&Self, StorageError> {
        let expected = PAGE_SIZE * 2;
        if bytes.len() < expected {
            return Err(StorageError::BufferTooSmall {
                expected,
                actual: bytes.len(),
            });
        }

        let addr = bytes.as_ptr() as usize;
        if addr % PAGE_SIZE != 0 {
            return Err(StorageError::MisalignedBuffer {
                address: addr,
                required_alignment: PAGE_SIZE,
            });
        }

        FromBytes::ref_from_prefix(bytes)
            .map(|(node, _)| node)
            .map_err(|e| StorageError::CastError(format!("{:?}", e)))
    }

    #[inline]
    pub fn from_bytes_mut(bytes: &mut [u8]) -> Result<&mut Self, StorageError> {
        let expected = PAGE_SIZE * 2;
        if bytes.len() < expected {
            return Err(StorageError::BufferTooSmall {
                expected,
                actual: bytes.len(),
            });
        }

        let addr = bytes.as_ptr() as usize;
        if addr % PAGE_SIZE != 0 {
            return Err(StorageError::MisalignedBuffer {
                address: addr,
                required_alignment: PAGE_SIZE,
            });
        }

        FromBytes::mut_from_prefix(bytes)
            .map(|(node, _)| node)
            .map_err(|e| StorageError::CastError(format!("{:?}", e)))
    }
}

/// A heap-allocated memory buffer guaranteed to be aligned to 4,096-byte page boundaries.
///
/// Designed specifically for zero-copy Direct I/O (`O_DIRECT`, `io_uring`) NVMe operations.
pub struct AlignedSectorBuffer {
    ptr: *mut u8,
    layout: Layout,
    len: usize,
}

impl AlignedSectorBuffer {
    /// Allocates an aligned buffer spanning `num_pages` 4 KB sectors.
    pub fn new(num_pages: usize) -> Result<Self, StorageError> {
        if num_pages == 0 {
            return Err(StorageError::BufferTooSmall {
                expected: PAGE_SIZE,
                actual: 0,
            });
        }

        let len = num_pages * PAGE_SIZE;
        let layout = Layout::from_size_align(len, PAGE_SIZE)
            .map_err(|e| StorageError::AllocationError(e.to_string()))?;

        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(StorageError::AllocationError("Memory allocation returned null".to_string()));
        }

        Ok(Self { ptr, layout, len })
    }

    /// Casts this aligned buffer directly into a `&DiskNode` with zero heap allocations.
    #[inline]
    pub fn as_disk_node(&self) -> Result<&DiskNode, StorageError> {
        DiskNode::from_bytes(self.as_ref())
    }

    /// Casts this aligned buffer directly into a `&mut DiskNode` with zero heap allocations.
    #[inline]
    pub fn as_disk_node_mut(&mut self) -> Result<&mut DiskNode, StorageError> {
        DiskNode::from_bytes_mut(self.as_mut())
    }

    /// Casts this aligned buffer directly into a `&DiskNode1536` with zero heap allocations.
    #[inline]
    pub fn as_disk_node_1536(&self) -> Result<&DiskNode1536, StorageError> {
        DiskNode1536::from_bytes(self.as_ref())
    }

    /// Casts this aligned buffer directly into a `&mut DiskNode1536` with zero heap allocations.
    #[inline]
    pub fn as_disk_node_1536_mut(&mut self) -> Result<&mut DiskNode1536, StorageError> {
        DiskNode1536::from_bytes_mut(self.as_mut())
    }
}

impl Deref for AlignedSectorBuffer {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl DerefMut for AlignedSectorBuffer {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl AsRef<[u8]> for AlignedSectorBuffer {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.deref()
    }
}

impl AsMut<[u8]> for AlignedSectorBuffer {
    #[inline]
    fn as_mut(&mut self) -> &mut [u8] {
        self.deref_mut()
    }
}

impl Drop for AlignedSectorBuffer {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr, self.layout);
        }
    }
}

unsafe impl Send for AlignedSectorBuffer {}
unsafe impl Sync for AlignedSectorBuffer {}
