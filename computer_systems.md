# Operating System

## Stack vs Heap

Stack: region of memory that stores temporary variables
- LIFO structure
- Fast access
- Managed automatically (grows/shrinks with function calls)
- Fixed size (stack overflow if exceeded)

Heap: region of memory used for dynamic memory allocation
- Program manually allocates/frees memory (or language runtime/GC does it)
- Slower access (pointer indirection, cache misses)
- Managed explicitly
- Larger than stack, can grow dynamically

## Process vs Thread

Process: a program in execution
- Each process has a separate memory space (code, data, heap, stack)
- Processes communicate via inter-process communication (IPC): pipes, sockets, shared memory, signals
- OS performs a full context switch when switching between processes (saves/restores full state)
- Managed by the process control block (PCB): created and updated by the OS for each process

Thread: smallest unit of execution within a process
- Multiple threads can exist in a single process
- Each thread has its own: stack, registers, program counter
- Threads share: heap, global variables, file descriptors, code
- Context switch between threads is cheaper than between processes
- User-level threads: managed by a user-space library, OS unaware; fast but can't truly parallelize
- Kernel-level threads: managed by the OS; can run on multiple cores simultaneously

## Concurrency

**Race condition**: two or more threads access shared data concurrently, and the outcome depends on execution order\
**Critical section**: code segment that accesses shared resources and must not be executed by more than one thread at a time

**Mutual exclusion mechanisms:**
- **Mutex**: binary lock; only one thread can hold it at a time; other threads block
- **Spinlock**: busy-waits instead of sleeping; low latency for short critical sections, wastes CPU if held long
- **Semaphore**: counter-based; binary semaphore (0/1) for mutual exclusion, counting semaphore for resource pools

**Deadlock**: set of threads where each is waiting for a resource held by another
- Four necessary conditions (Coffman conditions):
    1. Mutual exclusion: resource held exclusively
    2. Hold and wait: thread holds a resource while waiting for another
    3. No preemption: resources cannot be forcibly taken
    4. Circular wait: cycle of threads each waiting on the next
- Prevention: eliminate one of the four conditions
- Avoidance: Banker's algorithm — only grant requests that keep the system in a safe state
- Detection + recovery: detect cycles in resource allocation graph; kill or roll back a thread

**Livelock**: threads keep changing state in response to each other but make no progress\
**Starvation**: a thread is perpetually denied a resource it needs

## Memory

**Process address space layout** (low → high):
- **Text**: compiled code (read-only)
- **Data**: initialized global/static variables
- **BSS**: uninitialized global/static variables (zero-initialized)
- **Heap**: dynamic allocation, grows upward
- **Stack**: function call frames, grows downward

**Memory allocation:**
- `malloc`/`free` (C) or `new`/`delete` (C++) manage heap memory
- **Internal fragmentation**: allocated block is larger than requested
- **External fragmentation**: free blocks exist but are too scattered to satisfy a large request
- Allocators (e.g. jemalloc, tcmalloc) use free lists, slab allocation, and pooling to reduce fragmentation

**Garbage collection:**
- Runtime automatically reclaims unreachable memory
- Common strategies: reference counting (Python), mark-and-sweep (Java/Go), generational GC

## Cache

**Motivation**: CPU registers (~1 cycle) >> L1 cache (~4 cycles) >> L2 (~12 cycles) >> L3 (~40 cycles) >> RAM (~100+ cycles)

**Structure:**
- **Cache line**: smallest unit of data transferred between cache and memory (typically 64 bytes)
- **Cache hit**: data found in cache; **cache miss**: data must be fetched from slower memory
- **L1**: smallest, fastest, per-core; **L2**: larger, per-core; **L3**: shared across cores

**Replacement policies** (which line to evict on a miss):
- **LRU** (Least Recently Used): evict the line not used for the longest time
- **LFU** (Least Frequently Used): evict the line used least often
- **FIFO**: evict the oldest loaded line

**Write policies:**
- **Write-through**: write to cache and memory simultaneously; simple but slower
- **Write-back**: write only to cache; flush to memory when evicted; faster but more complex

**Cache coherence**: in multi-core systems, each core has its own L1/L2; coherence protocols (e.g. MESI) ensure all cores see a consistent view of shared memory

**Locality:**
- **Temporal locality**: recently accessed data is likely to be accessed again soon
- **Spatial locality**: data near recently accessed data is likely to be accessed soon (motivates prefetching and cache lines)

## Virtual vs Physical Memory

**Motivation**: give each process the illusion of a large, private address space regardless of actual RAM size

**Paging:**
- Virtual address space divided into fixed-size **pages** (typically 4 KB)
- Physical memory divided into same-size **frames**
- OS maintains a **page table** per process mapping virtual page → physical frame
- Only pages currently in use need to be in RAM

**Page fault**: CPU accesses a page not currently in RAM
- OS pauses the process, loads the page from disk (swap space), updates the page table, resumes

**TLB (Translation Lookaside Buffer)**: hardware cache of recent page table entries; avoids full page table walk on every memory access

**Segmentation**: divide address space into variable-size segments (code, stack, heap); less common than paging today

**Swapping**: move entire processes or pages to disk when RAM is full; allows overcommitting memory at the cost of performance

## Scheduler

**Goal**: efficiently multiplex CPU time across processes/threads

**Preemptive**: OS can interrupt a running process and switch to another\
**Non-preemptive (cooperative)**: process runs until it voluntarily yields or blocks

**Scheduling algorithms:**
- **FCFS** (First Come First Served): simple, no starvation, poor for short jobs behind long ones
- **SJF** (Shortest Job First): optimal average wait time; requires knowing burst time in advance
- **Round Robin**: each process gets a fixed time quantum; fair, good for interactive workloads
- **Priority Scheduling**: highest-priority process runs first; can cause starvation (fix: aging)
- **Multilevel Queue**: separate queues for different process types (e.g. interactive vs batch), each with its own algorithm

**Metrics:**
- **Throughput**: processes completed per unit time
- **Turnaround time**: total time from submission to completion
- **Waiting time**: time spent waiting in the ready queue
- **Response time**: time from submission to first response (important for interactive processes)

**Context switch**: save state of running process (registers, PC, etc.) to PCB; restore state of next process; has overhead (time, cache invalidation)

## File System

**Inode**: data structure storing file metadata
- Permissions, owner, timestamps, file size
- Pointers to data blocks (direct, indirect, double-indirect)
- Does **not** store the filename

**Directory**: maps filenames → inode numbers

**Common file systems:**
- **ext4**: default Linux FS; journaled, supports large files
- **NTFS**: Windows; journaling, ACLs, compression
- **FAT32**: simple, widely compatible; no permissions, 4 GB file size limit
- **APFS**: macOS/iOS; copy-on-write, snapshots, encryption

**Journaling**: write-ahead log records pending changes before applying them; allows recovery after a crash without full disk scan

**Hard link**: directory entry pointing to an inode; inode deleted only when all hard links removed\
**Soft (symbolic) link**: file containing a path to another file; can cross file systems; breaks if target is deleted

## I/O

**Polling (busy-wait)**: CPU repeatedly checks device status register; simple but wastes CPU cycles

**Interrupts**: device signals CPU (via interrupt line) when operation completes; CPU pauses current work, runs interrupt handler, resumes; more efficient than polling

**DMA (Direct Memory Access)**: device controller transfers data directly to/from memory without CPU involvement; CPU only involved at start and end of transfer; frees CPU for other work

**Buffering**: data held in memory while transferred between devices of different speeds\
**Caching**: frequently accessed disk data kept in memory to avoid re-reading\
**Spooling**: output queued to a buffer (e.g. print spooler) so the producing process can continue

## RAID

Redundant Array of Independent Disks — distribute data across multiple drives for performance and/or redundancy.

| Level  | Method                         | Min Drives | Fault Tolerance | Usable Capacity     |
| ------ | ------------------------------ | ---------- | --------------- | ------------------- |
| RAID 0 | Striping only                  | 2          | None            | 100%                |
| RAID 1 | Mirroring only                 | 2          | 1 drive         | 50%                 |
| RAID 5 | Striping + distributed parity  | 3          | 1 drive         | (N−1)/N             |
| RAID 6 | Striping + 2 parity blocks     | 4          | 2 drives        | (N−2)/N             |
| RAID 10| Mirrored pairs, then striped   | 4          | 1 per mirror pair| 50%                |

- RAID is not a backup — protects against drive failure, not accidental deletion or corruption

## Virtualization & Containers

**Virtualization**: emulate complete hardware; each VM runs its own OS kernel
- **Type 1 hypervisor** (bare-metal): runs directly on hardware; better performance (VMware ESXi, KVM, Hyper-V)
- **Type 2 hypervisor** (hosted): runs on top of a host OS (VirtualBox, VMware Workstation)

**Containers**: share the host OS kernel; isolate processes via Linux **namespaces** (PID, network, filesystem, etc.) and limit resources via **cgroups**
- **Docker**: container image = read-only layered filesystem (union mount); container = running image instance with a writable layer
- **Kubernetes**: orchestrates containers across a cluster (scheduling, scaling, health checks)

| Feature          | Virtual Machine | Container    |
| ---------------- | --------------- | ------------ |
| Isolation        | Strong (own OS) | Process-level|
| Startup time     | Minutes         | Milliseconds |
| Resource overhead| High            | Low          |
| Portability      | Limited         | High         |

# Compilers

## Preprocessor

Before code is compiled, the preprocessor performs file inclusion, macro expansion, and conditional compilation. Used in languages like C and C++.

For example, `#include <stdio.h>` is replaced with the textual content of `stdio.h`.

In C/C++, angle brackets `<>` search the standard compiler include paths; quotes `""` also search the current source file's directory.

## Lexical Analysis

**Input:** source code (or preprocessor output)
- Scans left to right (may require lookahead)
- Groups characters into **lexemes**, each corresponding to a **token**
- Ignores whitespace and comments
- Token types:
    - Identifiers (variable names)
    - Keywords (`int`, `if`, `return`)
    - Operators (`+`, `<`, `=`)
    - Literals (`true`, `1`, `"Hello"`)
    - Separators (`()`, `{}`, `;`)

**Output:** stream of tokens

## Syntax Analysis (Parsing)

**Input:** token stream
- Builds a **parse tree** (hierarchical representation of code structure)
- Detects syntax errors:
    - Mismatched parentheses
    - Missing semicolons
    - Invalid expression structure (e.g. `x 3` — missing operator)
- Compresses parse tree into an **abstract syntax tree (AST)**

**Output:** AST

## Semantic Analysis

**Input:** AST
- Verifies the tree is correct according to the language's semantic rules
- Detects semantic errors:
    - Undeclared variables
    - Invalid type conversions
    - Wrong number of function arguments
- Annotates the AST with type information and builds a symbol table
- Generates **intermediate representation (IR)**

**Output:** IR (e.g. three-address code, SSA form)

## Optimization

**Input:** IR

Machine-independent optimizations:
- **Constant folding/propagation**: evaluate constant expressions at compile time (`2 * 3` → `6`)
- **Dead code elimination**: remove code whose result is never used
- **Common subexpression elimination**: compute repeated expressions once and reuse the result
- **Loop invariant code motion**: move computations that don't change inside a loop to before it
- **Inlining**: replace a function call with the function body
- **Tail call optimization**: convert tail-recursive calls into iteration

Machine-dependent optimizations:
- **Register allocation**: assign frequently used values to registers (graph coloring)
- **Instruction scheduling**: reorder instructions to avoid pipeline stalls
- **Loop unrolling**: replicate loop body to reduce loop overhead
- **SIMD vectorization**: use vector instructions to process multiple data elements in parallel

**Output:** optimized IR

## Code Generation

**Input:** optimized IR

- **Instruction selection**: map IR operations to target ISA instructions
- **Register allocation**: assign virtual registers to physical registers; values that don't fit are *spilled* to the stack
- **Instruction scheduling**: reorder instructions to maximize pipeline throughput and hide memory latency
- Produces assembly or object code (`.o` files)
- **Linker** combines object files and resolves symbol references into a final executable

**Output:** target machine code or assembly

# Security

## Common Vulnerabilities

- **Buffer overflow**: writing past the end of a fixed-size buffer overwrites adjacent memory (e.g. return address), enabling arbitrary code execution; mitigations: bounds checking, stack canaries, ASLR, NX/DEP
- **SQL injection**: unsanitized user input is interpreted as SQL; mitigation: parameterized queries / prepared statements
- **XSS** (Cross-Site Scripting): attacker injects malicious scripts into a web page viewed by other users; mitigation: output encoding, Content Security Policy (CSP)
- **CSRF** (Cross-Site Request Forgery): tricks an authenticated user's browser into making an unintended request; mitigation: CSRF tokens, SameSite cookies
- **TOCTOU** (Time-of-Check-Time-of-Use): race condition between checking a condition and acting on it (e.g. checking file permissions then opening the file)

## Cryptography

- **Symmetric encryption**: same key for encrypt and decrypt (e.g. AES-256); fast, but key distribution is a challenge
- **Asymmetric encryption**: public key encrypts, private key decrypts (e.g. RSA, ECC); slow; used for key exchange and digital signatures
- **Hashing**: one-way function producing a fixed-length digest (e.g. SHA-256 for integrity, bcrypt/Argon2 for passwords); cannot be reversed
- **TLS/SSL**: asymmetric crypto used in the handshake to authenticate and exchange a symmetric session key; subsequent data encrypted with that session key

## Authentication & Authorization

- **Authentication**: verify identity — "who are you?" (password, MFA, certificate)
- **Authorization**: verify permissions — "what can you do?" (ACLs, RBAC, OAuth scopes)
- Common schemes: passwords, MFA (TOTP/FIDO2), OAuth2 (delegated authorization), JWT (stateless token), API keys
- **Principle of least privilege**: grant only the minimum permissions needed to perform a task

# Networks

## OSI Model

7 layers (mnemonic: "Please Do Not Throw Sausage Pizza Away"):

| Layer | Name           | Examples                        |
| ----- | -------------- | ------------------------------- |
| 7     | Application    | HTTP, DNS, SMTP, FTP            |
| 6     | Presentation   | TLS/SSL, JPEG, encoding         |
| 5     | Session        | session management, RPC         |
| 4     | Transport      | TCP, UDP                        |
| 3     | Network        | IP, ICMP, routing               |
| 2     | Data Link      | Ethernet, Wi-Fi, MAC addresses  |
| 1     | Physical       | cables, radio, bits on the wire |

## TCP/IP Model

Practical 4-layer model used in practice:

Link (Ethernet, Wi-Fi) → Internet (IP) → Transport (TCP/UDP) → Application (HTTP, DNS, ...)

## TCP vs UDP

|                 | TCP                                  | UDP                          |
| --------------- | ------------------------------------ | ---------------------------- |
| Connection      | Connection-oriented (3-way handshake) | Connectionless              |
| Reliability     | Guaranteed delivery, ordering, no duplication | No guarantees        |
| Flow/congestion | Congestion control, flow control     | None                         |
| Speed           | Slower                               | Faster                       |
| Use cases       | HTTP, email, file transfer           | DNS, video streaming, gaming |

**TCP 3-way handshake**: SYN → SYN-ACK → ACK

## HTTP / HTTPS

- HTTP: stateless application-layer protocol; client sends requests, server sends responses
- **Methods**: `GET` (read), `POST` (create/submit), `PUT` (replace), `PATCH` (partial update), `DELETE`
- **Status codes**: 1xx informational, 2xx success, 3xx redirect, 4xx client error, 5xx server error
- **HTTP/1.1**: persistent connections, pipelining
- **HTTP/2**: multiplexed streams over one connection, header compression, server push
- **HTTPS** = HTTP over TLS; encrypts and authenticates the connection

## DNS

Domain Name System: translates human-readable domain names to IP addresses

- **Resolution order**: local cache → OS resolver → recursive resolver → root nameserver → TLD nameserver → authoritative nameserver
- **Common record types:**
    - `A`: domain → IPv4 address
    - `AAAA`: domain → IPv6 address
    - `CNAME`: alias → canonical name
    - `MX`: mail exchanger for a domain
    - `TXT`: arbitrary text (used for SPF, DKIM, etc.)

## IP Addressing

- **IPv4**: 32-bit address, written as four octets (e.g. `192.168.1.1`); ~4.3 billion addresses
- **IPv6**: 128-bit address (e.g. `2001:db8::1`); effectively unlimited
- **CIDR notation**: `192.168.1.0/24` — first 24 bits are the network prefix, last 8 bits are host addresses
- **Private ranges** (not routable on the internet): `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`
- **NAT** (Network Address Translation): maps multiple private addresses to one public IP; enables private networks to share a public address

## Sockets

- Endpoint identified by (IP address, port, protocol)
- Well-known ports: 80 HTTP, 443 HTTPS, 22 SSH, 53 DNS
- **TCP server lifecycle**: `socket` → `bind` → `listen` → `accept` (blocks) → `read`/`write` → `close`
- **TCP client lifecycle**: `socket` → `connect` → `read`/`write` → `close`

# Database

## Relational vs NoSQL

|              | Relational (SQL)              | NoSQL                                        |
| ------------ | ----------------------------- | -------------------------------------------- |
| Schema       | Fixed, predefined             | Flexible / schemaless                        |
| Query        | SQL                           | Varies by type                               |
| ACID         | Typically full ACID           | Often eventual consistency                   |
| Scaling      | Vertical (primarily)          | Horizontal                                   |
| Examples     | PostgreSQL, MySQL, SQLite     | MongoDB (document), Redis (key-value), Cassandra (column), Neo4j (graph) |

## ACID Properties

- **Atomicity**: a transaction is all-or-nothing; if any part fails, the whole transaction is rolled back
- **Consistency**: a transaction brings the database from one valid state to another; constraints are never violated
- **Isolation**: concurrent transactions execute as if they were serial; one transaction's intermediate state is not visible to others
- **Durability**: a committed transaction persists even after a crash (written to disk/WAL)

## Indexes

Speed up read queries at the cost of write overhead and storage space.

- **B-tree index**: default in most databases; balanced tree; supports equality and range queries in O(log n)
- **Hash index**: O(1) average lookup; equality queries only; no range support
- **Composite index**: index on multiple columns; most selective column should come first; only useful when leading columns are queried
- **Covering index**: index includes all columns needed by a query, avoiding a table lookup entirely

## Transactions and Isolation Levels

Higher isolation → fewer anomalies → more locking overhead

| Level              | Dirty Read | Non-repeatable Read | Phantom Read |
| ------------------ | ---------- | ------------------- | ------------ |
| Read Uncommitted   | possible   | possible            | possible     |
| Read Committed     | prevented  | possible            | possible     |
| Repeatable Read    | prevented  | prevented           | possible     |
| Serializable       | prevented  | prevented           | prevented    |

- **Dirty read**: reading uncommitted data from another transaction
- **Non-repeatable read**: same row returns different values within one transaction
- **Phantom read**: a re-executed query returns different rows (another transaction inserted/deleted)

## Normalization

Process of structuring tables to reduce redundancy and dependency.

- **1NF**: each column holds atomic values; no repeating groups; each row is unique
- **2NF**: 1NF + no partial dependency (non-key column depends on the whole primary key, not just part of it)
- **3NF**: 2NF + no transitive dependency (non-key column does not depend on another non-key column)
- **Trade-off**: higher normal forms reduce redundancy and update anomalies but require more joins

## CAP Theorem

A distributed data store can guarantee at most **2 of 3**:
- **Consistency**: every read returns the most recent write
- **Availability**: every request receives a response (not necessarily the latest data)
- **Partition tolerance**: system continues operating despite network partitions

In practice, network partitions are unavoidable → must choose between **CP** (prioritize consistency, may reject requests) or **AP** (prioritize availability, may return stale data).

Examples: HBase is CP; Cassandra is AP; most traditional RDBMS are CA (single node, no partition tolerance).
