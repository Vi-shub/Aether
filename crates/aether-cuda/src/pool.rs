//! Memory pool for efficient GPU allocation.
//!
//! Memory pools reduce allocation overhead by reusing freed memory
//! instead of returning it to the CUDA driver.

use std::sync::Arc;
use std::collections::BTreeMap;
use parking_lot::Mutex;

use crate::device::CudaDevice;
use crate::memory::DeviceMemory;
use crate::error::{CudaError, Result};

/// Allocation statistics.
#[derive(Debug, Default, Clone)]
pub struct PoolStats {
    /// Total allocations.
    pub total_allocations: u64,
    /// Cache hits (reused memory).
    pub cache_hits: u64,
    /// Cache misses (new allocations).
    pub cache_misses: u64,
    /// Total bytes allocated.
    pub total_bytes_allocated: u64,
    /// Current bytes in use.
    pub bytes_in_use: u64,
    /// Current bytes cached (free but not returned to driver).
    pub bytes_cached: u64,
}

impl PoolStats {
    /// Cache hit rate.
    pub fn hit_rate(&self) -> f64 {
        if self.total_allocations == 0 {
            return 0.0;
        }
        self.cache_hits as f64 / self.total_allocations as f64
    }
}

/// Block in the memory pool.
struct PoolBlock {
    /// Underlying memory.
    memory: DeviceMemory,
    /// Whether this block is in use.
    in_use: bool,
}

/// GPU memory pool.
///
/// The pool maintains a cache of freed allocations, organized by size,
/// to quickly satisfy future allocations without going to the driver.
pub struct MemoryPool {
    /// Device this pool manages.
    device: Arc<CudaDevice>,
    /// Maximum pool size in bytes.
    max_size: u64,
    /// Minimum allocation size (smaller allocs are rounded up).
    min_alloc_size: u64,
    /// Free blocks by size (size -> list of blocks).
    free_blocks: Mutex<BTreeMap<u64, Vec<DeviceMemory>>>,
    /// Statistics.
    stats: Mutex<PoolStats>,
}

impl MemoryPool {
    /// Default minimum allocation size (256 KB).
    pub const DEFAULT_MIN_ALLOC: u64 = 256 * 1024;

    /// Create a new memory pool.
    pub fn new(device: Arc<CudaDevice>, max_size: u64) -> Self {
        Self {
            device,
            max_size,
            min_alloc_size: Self::DEFAULT_MIN_ALLOC,
            free_blocks: Mutex::new(BTreeMap::new()),
            stats: Mutex::new(PoolStats::default()),
        }
    }

    /// Create with custom minimum allocation size.
    pub fn with_min_alloc(device: Arc<CudaDevice>, max_size: u64, min_alloc: u64) -> Self {
        Self {
            device,
            max_size,
            min_alloc_size: min_alloc,
            free_blocks: Mutex::new(BTreeMap::new()),
            stats: Mutex::new(PoolStats::default()),
        }
    }

    /// Get device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Get maximum pool size.
    pub fn max_size(&self) -> u64 {
        self.max_size
    }

    /// Get statistics.
    pub fn stats(&self) -> PoolStats {
        self.stats.lock().clone()
    }

    /// Round size up to allocation granularity.
    fn round_size(&self, size: u64) -> u64 {
        let size = size.max(self.min_alloc_size);
        // Round up to next power of 2 for better reuse
        size.next_power_of_two()
    }

    /// Allocate from the pool.
    pub fn allocate(&self, size: u64) -> Result<PooledMemory> {
        let rounded_size = self.round_size(size);

        let mut stats = self.stats.lock();
        stats.total_allocations += 1;

        // Try to find a free block
        let mut free_blocks = self.free_blocks.lock();

        // Look for exact match or next larger size
        let mut found_size = None;
        for (&block_size, blocks) in free_blocks.range(rounded_size..) {
            if !blocks.is_empty() {
                found_size = Some(block_size);
                break;
            }
        }

        if let Some(block_size) = found_size {
            if let Some(blocks) = free_blocks.get_mut(&block_size) {
                if let Some(memory) = blocks.pop() {
                    stats.cache_hits += 1;
                    stats.bytes_in_use += memory.size();
                    stats.bytes_cached -= memory.size();

                    return Ok(PooledMemory {
                        memory: Some(memory),
                        requested_size: size,
                        pool: self,
                    });
                }
            }
        }

        drop(free_blocks);

        // No suitable free block, allocate new
        stats.cache_misses += 1;

        // Check if we have room
        if stats.bytes_in_use + rounded_size > self.max_size {
            // Try to free some cached memory
            drop(stats);
            self.trim(rounded_size)?;
            stats = self.stats.lock();
        }

        let memory = DeviceMemory::allocate(self.device.clone(), rounded_size)?;

        stats.total_bytes_allocated += rounded_size;
        stats.bytes_in_use += rounded_size;

        Ok(PooledMemory {
            memory: Some(memory),
            requested_size: size,
            pool: self,
        })
    }

    /// Return memory to the pool.
    fn release(&self, memory: DeviceMemory) {
        let size = memory.size();

        let mut stats = self.stats.lock();
        stats.bytes_in_use -= size;
        stats.bytes_cached += size;

        let mut free_blocks = self.free_blocks.lock();
        free_blocks.entry(size).or_default().push(memory);
    }

    /// Trim cached memory to make room.
    fn trim(&self, needed: u64) -> Result<()> {
        let mut free_blocks = self.free_blocks.lock();
        let mut stats = self.stats.lock();

        let mut freed = 0u64;

        // Free largest blocks first
        while freed < needed {
            // Find the largest size with blocks
            let largest = free_blocks
                .iter()
                .rev()
                .find(|(_, blocks)| !blocks.is_empty())
                .map(|(&size, _)| size);

            if let Some(size) = largest {
                if let Some(blocks) = free_blocks.get_mut(&size) {
                    if let Some(memory) = blocks.pop() {
                        freed += memory.size();
                        stats.bytes_cached -= memory.size();
                        // memory is dropped here, returning to driver
                    }
                }
            } else {
                break; // No more cached memory
            }
        }

        if freed < needed {
            return Err(CudaError::PoolError(format!(
                "Could not free enough memory: needed {}, freed {}",
                needed, freed
            )));
        }

        Ok(())
    }

    /// Clear all cached memory.
    pub fn clear_cache(&self) {
        let mut free_blocks = self.free_blocks.lock();
        let mut stats = self.stats.lock();

        for (_, blocks) in free_blocks.iter_mut() {
            for memory in blocks.drain(..) {
                stats.bytes_cached -= memory.size();
            }
        }

        free_blocks.clear();
    }

    /// Get current memory usage.
    pub fn memory_usage(&self) -> (u64, u64) {
        let stats = self.stats.lock();
        (stats.bytes_in_use, stats.bytes_cached)
    }
}

impl std::fmt::Debug for MemoryPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stats = self.stats.lock();
        f.debug_struct("MemoryPool")
            .field("device", &self.device.index())
            .field("max_size", &self.max_size)
            .field("in_use", &stats.bytes_in_use)
            .field("cached", &stats.bytes_cached)
            .finish()
    }
}

// =============================================================================
// Pooled Memory
// =============================================================================

/// Memory allocation from a pool.
///
/// When dropped, the memory is returned to the pool for reuse.
pub struct PooledMemory<'a> {
    memory: Option<DeviceMemory>,
    requested_size: u64,
    pool: &'a MemoryPool,
}

impl<'a> PooledMemory<'a> {
    /// Get the requested size.
    pub fn requested_size(&self) -> u64 {
        self.requested_size
    }

    /// Get the actual allocated size (may be larger due to rounding).
    pub fn allocated_size(&self) -> u64 {
        self.memory.as_ref().map(|m| m.size()).unwrap_or(0)
    }

    /// Get underlying memory.
    pub fn memory(&self) -> Option<&DeviceMemory> {
        self.memory.as_ref()
    }

    /// Get mutable underlying memory.
    pub fn memory_mut(&mut self) -> Option<&mut DeviceMemory> {
        self.memory.as_mut()
    }

    /// Take ownership of the memory (removes from pool tracking).
    pub fn take(mut self) -> Option<DeviceMemory> {
        self.memory.take()
    }
}

impl<'a> Drop for PooledMemory<'a> {
    fn drop(&mut self) {
        if let Some(memory) = self.memory.take() {
            self.pool.release(memory);
        }
    }
}

impl<'a> std::fmt::Debug for PooledMemory<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PooledMemory")
            .field("requested_size", &self.requested_size)
            .field("allocated_size", &self.allocated_size())
            .finish()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_allocate() {
        let device = Arc::new(CudaDevice::new(0).unwrap());
        let pool = MemoryPool::new(device, 1024 * 1024 * 1024); // 1GB

        let mem = pool.allocate(1024).unwrap();
        assert!(mem.allocated_size() >= 1024);

        let stats = pool.stats();
        assert_eq!(stats.total_allocations, 1);
        assert_eq!(stats.cache_misses, 1);
    }

    #[test]
    fn test_pool_reuse() {
        let device = Arc::new(CudaDevice::new(0).unwrap());
        let pool = MemoryPool::new(device, 1024 * 1024 * 1024);

        // Allocate and release
        let mem1 = pool.allocate(1024 * 1024).unwrap(); // 1MB
        let size = mem1.allocated_size();
        drop(mem1);

        // Should reuse
        let mem2 = pool.allocate(1024 * 1024).unwrap();
        assert_eq!(mem2.allocated_size(), size);

        let stats = pool.stats();
        assert_eq!(stats.total_allocations, 2);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 1);
    }

    #[test]
    fn test_pool_clear_cache() {
        let device = Arc::new(CudaDevice::new(0).unwrap());
        let pool = MemoryPool::new(device, 1024 * 1024 * 1024);

        let mem = pool.allocate(1024 * 1024).unwrap();
        drop(mem);

        let (in_use, cached) = pool.memory_usage();
        assert_eq!(in_use, 0);
        assert!(cached > 0);

        pool.clear_cache();

        let (in_use, cached) = pool.memory_usage();
        assert_eq!(in_use, 0);
        assert_eq!(cached, 0);
    }

    #[test]
    fn test_pool_hit_rate() {
        let device = Arc::new(CudaDevice::new(0).unwrap());
        let pool = MemoryPool::new(device, 1024 * 1024 * 1024);

        // First allocation - miss
        let m1 = pool.allocate(1024).unwrap();
        drop(m1);

        // Second allocation - hit
        let m2 = pool.allocate(1024).unwrap();
        drop(m2);

        // Third allocation - hit
        let _m3 = pool.allocate(1024).unwrap();

        let stats = pool.stats();
        assert_eq!(stats.total_allocations, 3);
        assert_eq!(stats.cache_hits, 2);
        assert!((stats.hit_rate() - 0.666).abs() < 0.01);
    }
}
