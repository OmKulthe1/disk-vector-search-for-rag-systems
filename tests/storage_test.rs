use lightvector::storage::{
    AlignedSectorBuffer, DiskNode, DiskNode1536, StorageError, MAX_DIM_1536, MAX_DIM_SINGLE_PAGE,
    MAX_NEIGHBORS, PAGE_SIZE,
};
use std::fs::{self, File};
use std::io::{Read, Write};

#[test]
fn test_disk_node_alignment_and_sizes() {
    // 1. Single-page DiskNode
    assert_eq!(std::mem::align_of::<DiskNode>(), 4096);
    assert_eq!(std::mem::size_of::<DiskNode>(), 4096);
    assert_eq!(std::mem::size_of::<DiskNode>() % 4096, 0);

    // 2. Dual-page DiskNode1536
    assert_eq!(std::mem::align_of::<DiskNode1536>(), 4096);
    assert_eq!(std::mem::size_of::<DiskNode1536>(), 8192);
    assert_eq!(std::mem::size_of::<DiskNode1536>() % 4096, 0);
}

#[test]
fn test_zero_copy_aligned_buffer_casting() {
    let mut buffer = AlignedSectorBuffer::new(1).unwrap();
    assert_eq!(buffer.len(), PAGE_SIZE);
    assert_eq!(buffer.as_ptr() as usize % PAGE_SIZE, 0);

    // Cast buffer directly into &mut DiskNode without any allocation
    {
        let node_mut = buffer.as_disk_node_mut().unwrap();
        node_mut.node_id = 42;
        let test_vec = [1.5f32, 2.5, 3.5, 4.5];
        let test_neighbors = [10u32, 20, 30];
        node_mut.set_vector(&test_vec).unwrap();
        node_mut.set_neighbors(&test_neighbors).unwrap();
    }

    // Cast buffer directly into immutable &DiskNode and verify fields
    let node = buffer.as_disk_node().unwrap();
    assert_eq!(node.node_id, 42);
    assert_eq!(node.vector_dim, 4);
    assert_eq!(node.vector(), &[1.5f32, 2.5, 3.5, 4.5]);
    assert_eq!(node.num_neighbors, 3);
    assert_eq!(node.neighbors(), &[10u32, 20, 30]);
}

#[test]
fn test_misalignment_and_size_validations() {
    // 1. Buffer too small
    let small_buffer = vec![0u8; 100];
    assert!(matches!(
        DiskNode::from_bytes(&small_buffer),
        Err(StorageError::BufferTooSmall { .. })
    ));

    // 2. Misaligned buffer
    let unaligned_bytes = vec![0u8; PAGE_SIZE + 64];
    // Offset by 1 byte to guarantee misalignment
    let misaligned_slice = &unaligned_bytes[1..1 + PAGE_SIZE];
    if (misaligned_slice.as_ptr() as usize) % PAGE_SIZE != 0 {
        assert!(matches!(
            DiskNode::from_bytes(misaligned_slice),
            Err(StorageError::MisalignedBuffer { .. })
        ));
    }

    // 3. Vector capacity exceeded
    let mut node = DiskNode::default();
    let oversized_vec = vec![1.0f32; MAX_DIM_SINGLE_PAGE + 1];
    assert!(matches!(
        node.set_vector(&oversized_vec),
        Err(StorageError::VectorDimensionExceeded { .. })
    ));

    // 4. Neighbor capacity exceeded
    let oversized_neighbors = vec![1u32; MAX_NEIGHBORS + 1];
    assert!(matches!(
        node.set_neighbors(&oversized_neighbors),
        Err(StorageError::NeighborsExceeded { .. })
    ));
}

#[test]
fn test_roundtrip_disk_serialization_single_page() {
    let test_file_path = "target/test_node_single_page.bin";

    let test_vec: Vec<f32> = (0..384).map(|i| i as f32 * 0.125).collect();
    let test_neighbors: Vec<u32> = (0..32).map(|i| i * 100).collect();

    let original_node = DiskNode::new(101, &test_vec, &test_neighbors).unwrap();
    assert_eq!(original_node.as_bytes().len(), PAGE_SIZE);

    // 1. Write node directly to disk file
    {
        let mut file = File::create(test_file_path).unwrap();
        file.write_all(original_node.as_bytes()).unwrap();
        file.flush().unwrap();
    }

    // 2. Read raw bytes back into AlignedSectorBuffer
    let mut read_buffer = AlignedSectorBuffer::new(1).unwrap();
    {
        let mut file = File::open(test_file_path).unwrap();
        file.read_exact(&mut read_buffer).unwrap();
    }

    // 3. Zero-copy cast and assert exact field identity
    let read_node = read_buffer.as_disk_node().unwrap();
    assert_eq!(read_node.node_id, 101);
    assert_eq!(read_node.vector_dim, 384);
    assert_eq!(read_node.vector(), test_vec.as_slice());
    assert_eq!(read_node.num_neighbors, 32);
    assert_eq!(read_node.neighbors(), test_neighbors.as_slice());

    // Assert raw byte identity
    assert_eq!(original_node.as_bytes(), read_node.as_bytes());

    // Clean up
    let _ = fs::remove_file(test_file_path);
}

#[test]
fn test_roundtrip_disk_serialization_dual_page_1536() {
    let test_file_path = "target/test_node_dual_page_1536.bin";

    let test_vec: Vec<f32> = (0..MAX_DIM_1536).map(|i| (i as f32).sin()).collect();
    let test_neighbors: Vec<u32> = (0..MAX_NEIGHBORS).map(|i| i as u32 + 500).collect();

    let original_node = DiskNode1536::new(202, &test_vec, &test_neighbors).unwrap();
    assert_eq!(original_node.as_bytes().len(), PAGE_SIZE * 2);

    // 1. Write node directly to disk file
    {
        let mut file = File::create(test_file_path).unwrap();
        file.write_all(original_node.as_bytes()).unwrap();
        file.flush().unwrap();
    }

    // 2. Read raw bytes back into AlignedSectorBuffer (2 pages = 8 KB)
    let mut read_buffer = AlignedSectorBuffer::new(2).unwrap();
    {
        let mut file = File::open(test_file_path).unwrap();
        file.read_exact(&mut read_buffer).unwrap();
    }

    // 3. Zero-copy cast and assert exact field identity
    let read_node = read_buffer.as_disk_node_1536().unwrap();
    assert_eq!(read_node.node_id, 202);
    assert_eq!(read_node.vector_dim, 1536);
    assert_eq!(read_node.vector(), test_vec.as_slice());
    assert_eq!(read_node.num_neighbors, 64);
    assert_eq!(read_node.neighbors(), test_neighbors.as_slice());

    // Assert complete bitwise identity
    assert_eq!(original_node.as_bytes(), read_node.as_bytes());

    // Clean up
    let _ = fs::remove_file(test_file_path);
}
