const builtin = @import("builtin");
const std = @import("std");
const Atomic = std.atomic.Atomic;

const Divider = @import("divide").Divider;

const TID = @import("../reclamation.zig").TID;

pub fn Domain(comptime T: type) type {
    return struct {
        // -- Fields --

        allocator: std.mem.Allocator,
        block_allocator: std.mem.Allocator,
        max_task_count: usize,
        max_he_count: usize,
        epoch_freq: u64,
        freq: u64,

        epoch_freq_divider: Divider(u64),
        freq_divider: Divider(u64),

        epoch: Atomic(u64) align(std.atomic.cache_line),

        task_local: []align(std.atomic.cache_line) TaskLocal,

        // -- Constants --

        const infinite_epoch: u64 = 0;

        // -- Types --

        const TaskLocal = struct {
            reservations: [*]Atomic(u64),
            retire_counter: u64,
            alloc_counter: u64,
            retired: std.TailQueue(Block),
        };

        const Block = struct {
            birth_epoch: u64,
            retire_epoch: u64,
            value: T,
        };

        const BlockNode = std.TailQueue(Block).Node;

        // -- Public functions --

        pub fn init(
            allocator: std.mem.Allocator,
            max_task_count: u32,
            max_he_count: usize,
            epoch_freq: u64,
            empty_freq: u64,
        ) !@This() {
            const tl = try allocator.alignedAlloc(TaskLocal, std.atomic.cache_line, max_task_count);

            var i: u32 = 0;
            errdefer {
                for (0..i) |j| {
                    allocator.free(tl[j].reservations[0..max_he_count]);
                }
                allocator.free(tl);
            }
            while (i < max_task_count) : (i += 1) {
                tl[i].reservations = (try allocator.alloc(Atomic(u64), max_he_count)).ptr;
                for (tl[i].reservations[0..max_he_count]) |*r| {
                    r.store(infinite_epoch, .Release);
                }
                tl[i].retire_counter = 0;
                tl[i].alloc_counter = 0;
                tl[i].retired = std.TailQueue(Block){};
            }

            const freq = epoch_freq + empty_freq;

            return .{
                .allocator = allocator,
                .block_allocator = allocator,
                .max_task_count = max_task_count,
                .max_he_count = max_he_count,
                .epoch_freq = epoch_freq,
                .freq = freq,

                .epoch_freq_divider = Divider(u64).init(epoch_freq * max_task_count),
                .freq_divider = Divider(u64).init(freq),

                .epoch = Atomic(u64).init(1),

                .task_local = tl,
            };
        }
        pub fn deinit(self: *@This()) void {
            for (self.task_local) |*tl| {
                // TODO(luis): soundness?
                while (tl.retired.popFirst()) |node| self.reclaimNode(node);
                self.allocator.free(tl.reservations[0..self.max_he_count]);
            }
            self.allocator.free(self.task_local);
        }

        pub fn alloc(self: *@This(), tid: TID) !*T {
            self.task_local[@intFromEnum(tid)].alloc_counter += 1;

            // Check that the alloc_counter is fully divisible by epoch_freq * max_task_count
            const should_advance_epoch = should_advance: {
                const q = self.epoch_freq_divider.divTrunc(self.task_local[@intFromEnum(tid)].alloc_counter);
                break :should_advance (q * self.epoch_freq * self.max_task_count) == self.task_local[@intFromEnum(tid)].alloc_counter;
            };
            if (should_advance_epoch) _ = self.epoch.fetchAdd(1, .AcqRel);

            // TODO(luis): if we fail to allocate is everything correct? or should we allocate first?
            const node = try self.block_allocator.create(BlockNode);
            node.data.birth_epoch = self.epoch.load(.Acquire);
            return &node.data.value;
        }

        pub fn read(self: *@This(), tid: TID, index: usize, atomic_ptr: *Atomic(?*T)) ?*T {
            const tl = &self.task_local[@intFromEnum(tid)];
            var prev_epoch = tl.reservations[index].load(.Acquire);
            while (true) {
                const ptr = atomic_ptr.load(.Acquire);
                const curr_epoch = self.epoch.load(.Acquire);
                if (curr_epoch == prev_epoch) {
                    return ptr;
                } else {
                    tl.reservations[index].store(curr_epoch, .Release);
                    prev_epoch = curr_epoch;
                }
            }
        }

        pub fn retire(self: *@This(), tid: TID, ptr: ?*T) void {
            if (ptr == null) return;

            const tl = &self.task_local[@intFromEnum(tid)];

            const brock = &tl.retired;

            const block = @fieldParentPtr(Block, "value", ptr.?);
            const node = @fieldParentPtr(BlockNode, "data", block);

            const retire_epoch = self.epoch.load(.Acquire);
            block.retire_epoch = retire_epoch;
            brock.append(node);

            // Check that the retire_counter is fully divisible by freq
            const should_cleanup = should_advance: {
                const q = self.freq_divider.divTrunc(tl.retire_counter);
                break :should_advance (q * self.freq) == tl.retire_counter;
            };
            if (should_cleanup) {
                if (self.epoch.load(.Acquire) == retire_epoch) _ = self.epoch.fetchAdd(1, .AcqRel);
                self.emptyTrash(tid);
            }
            tl.retire_counter += 1;
        }

        pub fn reclaim(self: *@This(), ptr: ?*T) void {
            if (ptr == null) return;

            const block = @fieldParentPtr(Block, "value", ptr.?);
            const node = @fieldParentPtr(BlockNode, "data", block);
            self.reclaimNode(node);
        }

        // -- Private functions --

        fn emptyTrash(self: *@This(), tid: TID) void {
            const brock = &self.task_local[@intFromEnum(tid)].retired;

            var cur = brock.first;
            while (cur) |node| {
                const next = node.next;
                defer cur = next;

                const block = &node.data;
                if (self.canDelete(block.birth_epoch, block.retire_epoch)) {
                    brock.remove(node);
                    self.reclaimNode(node);
                }
            }
        }

        fn canDelete(self: *const @This(), birth_epoch: u64, retire_epoch: u64) bool {
            for (self.task_local) |*tl| {
                var j: usize = 0;

                var is_current: u1 = 0;
                while (j < self.max_he_count) : (j += 1) {
                    const epoch = tl.reservations[j].load(.Acquire);

                    const cur = @intFromBool(birth_epoch <= epoch) & @intFromBool(epoch <= retire_epoch) & @intFromBool(epoch != infinite_epoch);
                    is_current = cur | is_current;
                }
                if (is_current != 0) return false;
            }
            return true;
        }

        fn reclaimNode(self: *@This(), node: *BlockNode) void {
            // TODO(luis): call some kind of destructor here
            self.block_allocator.destroy(node);
        }
    };
}
