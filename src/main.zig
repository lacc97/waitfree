const std = @import("std");

const reclamations = @import("primitives/reclamation.zig");

const arxiv1510_00116 = @import("arxiv1510_00116.zig");

const max_tasks = 8;
const max_he_count = 48;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){ .backing_allocator = std.heap.c_allocator };
    defer _ = gpa.deinit();

    const allocator = std.heap.c_allocator;

    var he = try reclamations.he.Domain(u32).init(
        allocator,
        max_tasks,
        max_he_count,
        10,
        10,
    );
    defer he.deinit();

    var ptrs: [max_he_count]std.atomic.Atomic(?*u32) = undefined;
    for (&ptrs) |*p| p.* = std.atomic.Atomic(?*u32).init(null);
    defer {
        for (&ptrs) |*p| he.reclaim(p.load(.Acquire));
    }

    var barrier = std.Thread.ResetEvent{};
    var threads: [max_tasks]std.Thread = undefined;
    {
        var i: u32 = 0;
        errdefer {
            barrier.set();
            for (0..i) |j| threads[j].join();
        }
        for (&threads) |*t| {
            t.* = try std.Thread.spawn(.{}, threadFn, .{
                @as(reclamations.TID, @enumFromInt(i)),
                &he,
                &barrier,
                &ptrs,
            });
            i += 1;
        }
    }
    barrier.set();
    for (&threads) |*t| t.join();
}

fn threadFn(
    tid: reclamations.TID,
    he: *reclamations.he.Domain(u32),
    barrier: *std.Thread.ResetEvent,
    ptrs: *[max_he_count]std.atomic.Atomic(?*u32),
) void {
    barrier.wait();

    var prng = std.rand.DefaultPrng.init(std.crypto.random.int(u64));
    const random = prng.random();

    for (0..10000) |_| {
        const idx = random.intRangeLessThan(usize, 0, max_he_count);
        if (random.boolean()) {
            const ptr = he.read(tid, idx, &ptrs.*[idx]);
            _ = ptr;
        } else {
            const old_ptr = ptrs.*[idx].swap(he.alloc(tid) catch return, .AcqRel);
            he.retire(tid, old_ptr);
        }
    }
}
