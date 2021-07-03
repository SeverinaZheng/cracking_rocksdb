//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//


#include "memtable/adaptive_linklist_rep.h"

#include <algorithm>
#include <atomic>
#include <vector>
#include <string.h>
#include <string>
#include "db/memtable.h"
#include "memory/arena.h"
#include "memtable/skiplist.h"
#include "monitoring/histogram.h"
#include "port/port.h"
#include "rocksdb/memtablerep.h"
#include "rocksdb/slice.h"
#include "rocksdb/slice_transform.h"
#include "util/hash.h"

namespace ROCKSDB_NAMESPACE {
namespace {

typedef const char* Key;
typedef SkipList<Key, const MemTableRep::KeyComparator&> MemtableSkipList;
typedef std::atomic<void*> Pointer;

// A data structure used as the header of a link list of a hash bucket.
struct BucketHeader {
  Pointer next;
  std::atomic<uint32_t> num_entries;
  Key endpoint_key;

  explicit BucketHeader(void* n, uint32_t count, Key key)
      : next(n), num_entries(count),endpoint_key(key) {}

  uint32_t GetNumEntries() const {
    return num_entries.load(std::memory_order_relaxed);
  }

  Key GetEndPointKey() const {
    return endpoint_key;
  }

  // REQUIRES: called from single-threaded Insert()
  void IncNumEntries() {
    // Only one thread can do write at one time. No need to do atomic
    // incremental. Update it with relaxed load and store.
    num_entries.store(GetNumEntries() + 1, std::memory_order_relaxed);
  }
};

struct Node {
  // Accessors/mutators for links.  Wrapped in methods so we can
  // add the appropriate barriers as necessary.
  Node* Next() {
    // Use an 'acquire load' so that we observe a fully initialized
    // version of the returned Node.
    return next_.load(std::memory_order_acquire);
  }
  void SetNext(Node* x) {
    // Use a 'release store' so that anybody who reads through this
    // pointer observes a fully initialized version of the inserted node.
    next_.store(x, std::memory_order_release);
  }
  // No-barrier variants that can be safely used in a few locations.
  Node* NoBarrier_Next() {
    return next_.load(std::memory_order_relaxed);
  }

  void NoBarrier_SetNext(Node* x) { next_.store(x, std::memory_order_relaxed); }

  // Needed for placement new below which is fine
  Node() {}

 private:
  std::atomic<Node*> next_;

  // Prohibit copying due to the below
  Node(const Node&) = delete;
  Node& operator=(const Node&) = delete;

 public:
  char* key;
};

// Memory structure of the mem table:
// It is a hash table, each bucket points to one entry, a linked list or a
// skip list. In order to track total number of records in a bucket to determine
// whether should switch to skip list, a header is added just to indicate
// number of entries in the bucket.
//
//
//          
//          
//          
//            +---> +-------+
//            |     | Next  +--> NULL
//            |     +-------+
//  +-----+   |     |       |  Case 1. only bucket headers(headers containing
//  |     +   |     | Data  |          endpoints which are in order)
//  +-----+   |     |       |          
//  +-----+   |     +-------+
//  |     +---+
//  +-----+     +-> +-------+  +> +-------+  +-> +-------+
//  |     |     |   | Next  +--+  | Next  +--+   | Next  +-->NULL
//  +-----+     |   +-------+     +-------+      +-------+
//  |     +-----+   | Count |     |       |      |       |
//  +-----+         +-------+     | Data  |      | Data  |
//  |     |                       |       |      |       |
//  +-----+                       |       |      |       |
//  |     |                       +-------+      +-------+
//  +-----+        

//
// We don't have data race when changing cases because:
// (1) When changing from case 2->3, we create a new bucket header, put the
//     single node there first without changing the original node, and do a
//     release store when changing the bucket pointer. In that case, a reader
//     who sees a stale value of the bucket pointer will read this node, while
//     a reader sees the correct value because of the release store.
// (2) When changing case 3->4, a new header is created with skip list points
//     to the data, before doing an acquire store to change the bucket pointer.
//     The old header and nodes are never changed, so any reader sees any
//     of those existing pointers will guarantee to be able to iterate to the
//     end of the linked list.
// (3) Header's next pointer in case 3 might change, but they are never equal
//     to itself, so no matter a reader sees any stale or newer value, it will
//     be able to correctly distinguish case 3 and 4.
//
// The reason that we use case 2 is we want to make the format to be efficient
// when the utilization of buckets is relatively low. If we use case 3 for
// single entry bucket, we will need to waste 12 bytes for every entry,
// which can be significant decrease of memory utilization.
class AdaptiveLinkListRep : public MemTableRep {
 public:
  AdaptiveLinkListRep(const MemTableRep::KeyComparator& compare,
                  Allocator* allocator, const SliceTransform* transform, 
                  //size_t bucket_size,uint32_t threshold_use_skiplist,//
                  size_t huge_page_tlb_size, Logger* logger,
                  int bucket_entries_logging_threshold,
                  bool if_log_bucket_dist_when_flash);

  void Insert(KeyHandle handle) override;

  bool Contains(const char* key) const override;

  size_t ApproximateMemoryUsage() override;

  void Get(const LookupKey& k, void* callback_args,
           bool (*callback_func)(void* arg, const char* entry)) override;

  ~AdaptiveLinkListRep() override;

  MemTableRep::Iterator* GetIterator(Arena* arena = nullptr) override;

  MemTableRep::Iterator* GetDynamicPrefixIterator(
      Arena* arena = nullptr) override;

  std::vector<Slice> GetEndpointList() override;

  void RebuildEndpointList(std::vector<Slice> endpoints) override;

  bool isAdaptive() override;

 private:
  friend class DynamicIterator;

  // Maps slices (which are transformed user keys) to buckets of keys sharing
  // the same transform.
  Pointer* buckets_;
  //read a key will be added to endpoint_list, which will take effect at next flush
  //and be the buckets'endpoints value after next flush
  MemtableSkipList* endpoint_list_;
  //length of next endpoint_list; a parameter for allocating memory for next flush
  size_t endpoint_num_;
  //the max index of buckets_
  size_t buckets_num_;


  // The user-supplied transform whose domain is the user keys.
  const SliceTransform* transform_;

  const MemTableRep::KeyComparator& compare_;

  Logger* logger_;
  size_t huge_page_tlb_size_;
  int bucket_entries_logging_threshold_;
  bool if_log_bucket_dist_when_flash_;

  bool LinkListContains(Node* head, const Slice& key) const;


  Node* GetLinkListFirstNode(Pointer* first_next_pointer) const;

  //former GetHash
  size_t FindRange(const Slice& slice) const {
    size_t low = 0;
    size_t high = buckets_num_;
    size_t middle ;
    while (low < high){
      middle = (low + high)/2;
      BucketHeader* header =  reinterpret_cast<BucketHeader*> (buckets_[middle].load(std::memory_order_acquire));
      Key endpoint = header->GetEndPointKey();
      if(endpoint == nullptr) return middle;
      auto diff = compare_(endpoint,slice);
      if ( diff == 0) return middle;
      else if (diff > 0){
          if(middle == 0) return middle;
          Key prev_endpoint = (reinterpret_cast<BucketHeader*>(buckets_[middle - 1].load(std::memory_order_acquire)))->GetEndPointKey();
        if (compare_(prev_endpoint,slice) < 0)  return middle;
        else  high = middle - 1;
      }else
        low = middle + 1;
    }
    return high;

  }

  Pointer* GetBucket(size_t i) const {
    return static_cast<Pointer*>(buckets_[i].load(std::memory_order_acquire));
  }

  Pointer* GetBucket(const Slice& slice) const {
    return GetBucket(FindRange(slice));
  }

  bool Equal(const Slice& a, const Key& b) const {
    return (compare_(b, a) == 0);
  }

  bool Equal(const Key& a, const Key& b) const { return (compare_(a, b) == 0); }

  Node* FindWetherInBucket(Node* head, const Slice& key) const;
  Node* FindLessOrEqualInBucket(Node* head, const Slice& key) const;

  class FullListIterator : public MemTableRep::Iterator {
   public:
    explicit FullListIterator(Node* head, Allocator* allocator)
        : head_(head), allocator_(allocator){
          node_ = head_;
        }

    ~FullListIterator() override {}

    // Returns true iff the iterator is positioned at a valid node.
    bool Valid() const override { return node_!= nullptr;}

    // Returns the key at the current position.
    // REQUIRES: Valid()
    const char* key() const override {
      assert(Valid());
      return node_->key;
    }

    // Advances to the next position.
    // REQUIRES: Valid()
    void Next() override {
      assert(Valid());
      node_ = node_->Next();
    }

    // Advances to the previous position.
    // REQUIRES: Valid()
    void Prev() override {
      Reset(nullptr);
    }

    // Advance to the first entry with a key >= target
    void Seek(const Slice& /*internal_key*/,
              const char* /*memtable_key*/) override {
      Reset(nullptr);
    }

    // Retreat to the last entry with a key <= target
    void SeekForPrev(const Slice& /*internal_key*/,
                     const char* /*memtable_key*/) override {
      Reset(nullptr);
    }

    // Position at the first entry in collection.
    // Final state of iterator is Valid() iff collection is not empty.
    void SeekToFirst() override { node_ = head_; }

    // Position at the last entry in collection.
    // Final state of iterator is Valid() iff collection is not empty.
    void SeekToLast() override { Reset(nullptr); }

   protected:
    void Reset(Node* head) {
      head_ = head;
      node_ = nullptr;
    }
   private:
    Node* head_;
    Node* node_;
    std::unique_ptr<Allocator> allocator_;
    std::string tmp_;       // For passing to EncodeKey
  };


  class LinkListIterator : public MemTableRep::Iterator {
   public:
    explicit LinkListIterator(const AdaptiveLinkListRep* const adaptive_link_list_rep,
                              Node* head)
        : adaptive_link_list_rep_(adaptive_link_list_rep),
          head_(head),
          node_(nullptr) {}

    ~LinkListIterator() override {}

    // Returns true iff the iterator is positioned at a valid node.
    bool Valid() const override { return node_ != nullptr; }

    // Returns the key at the current position.
    // REQUIRES: Valid()
    const char* key() const override {
      assert(Valid());  
      return node_->key;
    }

    // Advances to the next position.
    // REQUIRES: Valid()
    void Next() override {
      assert(Valid());
      node_ = node_->Next();
    }

    //get current node
    Node* GetNode(){
      return node_;
    }

    // Advances to the previous position.
    // REQUIRES: Valid()
    void Prev() override {
      // Prefix iterator does not support total order.
      // We simply set the iterator to invalid state
      Reset(nullptr);
    }

    // Advance to the first entry with a key >= target
    void Seek(const Slice& internal_key,
              const char* /*memtable_key*/) override {
      node_ = adaptive_link_list_rep_->FindWetherInBucket(head_,internal_key);
    }

    // Retreat to the last entry with a key <= target
    void SeekForPrev(const Slice& /*internal_key*/,
                     const char* /*memtable_key*/) override {
      // Since we do not support Prev()
      // We simply do not support SeekForPrev
      Reset(nullptr);
    }

    // Position at the first entry in collection.
    // Final state of iterator is Valid() iff collection is not empty.
    void SeekToFirst() override {
      // Prefix iterator does not support total order.
      // We simply set the iterator to invalid state
      Reset(nullptr);
    }

    // Position at the last entry in collection.
    // Final state of iterator is Valid() iff collection is not empty.
    void SeekToLast() override {
      // Prefix iterator does not support total order.
      // We simply set the iterator to invalid state
      Reset(nullptr);
    }
  
   void SeekToHead() {
      node_ = head_;
    }

   protected:
    void Reset(Node* head) {
      head_ = head;
      node_ = nullptr;
    }
   private:
    friend class AdapticeLinkListRep;
    const AdaptiveLinkListRep* const adaptive_link_list_rep_;
    Node* head_;
    Node* node_;

  };

  class DynamicIterator : public AdaptiveLinkListRep::LinkListIterator {
   public:
    explicit DynamicIterator(AdaptiveLinkListRep& memtable_rep)
        : AdaptiveLinkListRep::LinkListIterator(&memtable_rep, nullptr),
          memtable_rep_(memtable_rep) {}

    // Advance to the first entry with a key >= target
    void Seek(const Slice& k, const char* memtable_key) override {
      //auto transformed = memtable_rep_.GetPrefix(k);
      //auto* bucket = memtable_rep_.GetBucket(transformed);
      AdaptiveLinkListRep::LinkListIterator::Seek(k, memtable_key);
      
    }

    bool Valid() const override {
      return AdaptiveLinkListRep::LinkListIterator::Valid();
    }

    const char* key() const override {
      return AdaptiveLinkListRep::LinkListIterator::key();
    }

    void Next() override {
        AdaptiveLinkListRep::LinkListIterator::Next();
    }

   private:
    // the underlying memtable
    const AdaptiveLinkListRep& memtable_rep_;
    std::unique_ptr<MemtableSkipList::Iterator> skip_list_iter_;
  };

  class EmptyIterator : public MemTableRep::Iterator {
    // This is used when there wasn't a bucket. It is cheaper than
    // instantiating an empty bucket over which to iterate.
   public:
    EmptyIterator() { }
    bool Valid() const override { return false; }
    const char* key() const override {
      assert(false);
      return nullptr;
    }
    void Next() override {}
    void Prev() override {}
    void Seek(const Slice& /*user_key*/,
              const char* /*memtable_key*/) override {}
    void SeekForPrev(const Slice& /*user_key*/,
                     const char* /*memtable_key*/) override {}
    void SeekToFirst() override {}
    void SeekToLast() override {}

   private:
  };
};

AdaptiveLinkListRep::AdaptiveLinkListRep(
    const MemTableRep::KeyComparator& compare, Allocator* allocator,
    const SliceTransform* transform, 
    size_t huge_page_tlb_size, Logger* logger,
    int bucket_entries_logging_threshold, bool if_log_bucket_dist_when_flash)
    : MemTableRep(allocator),
      endpoint_num_(0),
      buckets_num_(0),
      // Threshold to use skip list doesn't make sense if less than 3, so we
      // force it to be minimum of 3 to simplify implementation.
      //threshold_use_skiplist_(std::max(threshold_use_skiplist, 3U)),
      transform_(transform),
      compare_(compare),
      logger_(logger),
      huge_page_tlb_size_(huge_page_tlb_size),
      bucket_entries_logging_threshold_(bucket_entries_logging_threshold),
      if_log_bucket_dist_when_flash_(if_log_bucket_dist_when_flash) {
  char* mem = allocator_->AllocateAligned(sizeof(Pointer),
                                      huge_page_tlb_size, logger);

  buckets_ = new (mem) Pointer[0];
  auto skipl = allocator_->AllocateAligned(sizeof(MemtableSkipList));
  //max height = 12 branch factor = 4
  endpoint_list_ = new (skipl) MemtableSkipList(compare_,allocator_,12,4);
  BucketHeader* header = nullptr;
  auto* mem2 = allocator_->AllocateAligned(sizeof(BucketHeader));
  header = new (mem2) BucketHeader(nullptr, 0,nullptr);
  buckets_[0].store(header, std::memory_order_relaxed);

}

AdaptiveLinkListRep::~AdaptiveLinkListRep() {
}

Node* AdaptiveLinkListRep::GetLinkListFirstNode(Pointer* first_next_pointer) const {
  if (first_next_pointer == nullptr) {
    return nullptr;
  }
  BucketHeader* header = reinterpret_cast<BucketHeader*>(first_next_pointer);
  return reinterpret_cast<Node*>(
        header->next.load(std::memory_order_acquire));
}

std::vector<Slice> AdaptiveLinkListRep::GetEndpointList(){
  MemtableSkipList::Iterator iter(endpoint_list_);
  std::vector<Slice> key_list;
  iter.SeekToFirst();
  for(;iter.Valid();iter.Next()){
    key_list.emplace_back(static_cast<const char*>(iter.key()));
  }
  return key_list;
}

void AdaptiveLinkListRep::RebuildEndpointList(std::vector<Slice> endpoints) {
  endpoint_num_ = endpoints.size();
  assert(endpoint_num_ != 0);
  char* mem = allocator_->AllocateAligned(sizeof(Pointer) * (endpoint_num_+1),
                                        huge_page_tlb_size_, logger_);

  buckets_ = new (mem) Pointer[endpoint_num_+1];
  for (size_t i = 0; i < endpoint_num_; ++i) {
    BucketHeader* header = nullptr;
    auto* mem2 = allocator_->AllocateAligned(sizeof(BucketHeader));
    Key k = endpoints[i].data();
    endpoint_list_->Insert(k);    
    header = new (mem2) BucketHeader(nullptr, 0,k);
    buckets_[i].store(header, std::memory_order_relaxed);
  }
  BucketHeader* header = nullptr;
  auto* mem2 = allocator_->AllocateAligned(sizeof(BucketHeader));
  header = new (mem2) BucketHeader(nullptr, 0,nullptr);
  buckets_[endpoint_num_].store(header, std::memory_order_relaxed);
  buckets_num_ = endpoint_num_;

  
  // test endpoints
  for (size_t i = 0 ; i <= endpoint_num_; i++){
     BucketHeader* headeri =  reinterpret_cast<BucketHeader*> (buckets_[i].load(std::memory_order_acquire));
     Key key_ = headeri->GetEndPointKey();
    printf("%s",key_);
  }
}

bool AdaptiveLinkListRep::isAdaptive(){return true;}

void AdaptiveLinkListRep::Insert(KeyHandle handle) {
  auto* key = static_cast<char*>(handle);
  //in order to get consistent with keys used in the get() function
  //the node's key will be raw and the key to find the buckets will 
  //be userkey
  Slice internal_key = GetLengthPrefixedSlice(key);
  Slice transformed = ExtractUserKey(internal_key);
  char* mem = allocator_->AllocateAligned(sizeof(Node) + strlen(transformed.data()));
  Node* x = new (mem) Node();
  x->key = key;
  x->NoBarrier_SetNext(nullptr);

  auto& bucket = buckets_[FindRange(transformed)];
  Pointer* first_next_pointer =
      static_cast<Pointer*>(bucket.load(std::memory_order_relaxed));

  assert(first_next_pointer != nullptr);
  BucketHeader* header = nullptr;
  header = reinterpret_cast<BucketHeader*>(first_next_pointer);
  if (bucket_entries_logging_threshold_ > 0 &&
      header->GetNumEntries() ==
          static_cast<uint32_t>(bucket_entries_logging_threshold_)) {
    Info(logger_, "AdaptiveLinkedList bucket %s" 
                  " has more than %d "
                  "entries. Key to insert: %s",
         header->GetEndPointKey(), header->GetNumEntries(),
         (x->key));
  }
    // Need to insert to the unsorted linked list at the first non-header element
  if(header->GetNumEntries() > 0){
    Node* first =
        reinterpret_cast<Node*>(header->next.load(std::memory_order_relaxed));
    assert(first != nullptr);
    // Advance counter unless the bucket needs to be advanced to skip list.
    // In that case, we need to make sure the previous count never exceeds
    // threshold_use_skiplist_ to avoid readers to cast to wrong format.
    Node* cur = first;

    // Our data structure does not allow duplicate insertion
    assert(cur == nullptr);

    // NoBarrier_SetNext() suffices since we will add a barrier when
    // we publish a pointer to "x" in prev[i].
    x->NoBarrier_SetNext(cur);

    }
    header->next.store(static_cast<void*>(x), std::memory_order_release);
    header->IncNumEntries();
    endpoint_list_->Insert(x->key);
    endpoint_num_++;
    //uint32_t nums = header->GetNumEntries();
    //Key k = header->GetEndPointKey();
  }


bool AdaptiveLinkListRep::Contains(const char* key) const {
  Slice transformed = Slice(key);
  auto bucket = GetBucket(transformed);
  if (bucket == nullptr) {
    return false;
  }
  Node* header = GetLinkListFirstNode(bucket);
  return LinkListContains(header, transformed);
  
}

size_t AdaptiveLinkListRep::ApproximateMemoryUsage() {
  // Memory is always allocated from the allocator.
  return 0;
}

void AdaptiveLinkListRep::Get(const LookupKey& k, void* callback_args,
                          bool (*callback_func)(void* arg, const char* entry)) {
  auto transformed = Slice(k.user_key());
  auto bucket = GetBucket(transformed);

  auto* link_list_head = GetLinkListFirstNode(bucket);
  if (link_list_head != nullptr) {
    LinkListIterator iter(this, link_list_head);
    for (iter.Seek(k.internal_key(), nullptr);
         iter.Valid() && callback_func(callback_args, iter.key());
         iter.Next()) {
    }
  }
  
}

MemTableRep::Iterator* AdaptiveLinkListRep::GetIterator(Arena* alloc_arena) {
  // allocate a new arena of similar size to the one currently in use
  Arena* new_arena = new Arena(allocator_->BlockSize());
  char* mem = allocator_->AllocateAligned(sizeof(Pointer));
  Pointer* all_entries_ = new (mem) Pointer[1];
  Node* cur = nullptr;
  Node* head = nullptr;
  HistogramImpl keys_per_bucket_hist;

  for (size_t i = 0; i < buckets_num_+1; ++i) {
    int count = 0;
    auto* bucket = GetBucket(i);
    if (bucket != nullptr) {
      auto* link_list_head = GetLinkListFirstNode(bucket);
      if (link_list_head != nullptr) {
        LinkListIterator itr(this, link_list_head);
        for (itr.SeekToHead(); itr.Valid(); itr.Next()) {
          Node* this_node = itr.GetNode();
          if(cur == nullptr){
            all_entries_[0].store(this_node,std::memory_order_relaxed);
            head = this_node;
          }else{
            cur->SetNext(this_node);
          }
          cur = this_node;
          count++;
        }
      }
      
    }
    if (if_log_bucket_dist_when_flash_) {
      keys_per_bucket_hist.Add(count);
    }
  }
  if (if_log_bucket_dist_when_flash_ && logger_ != nullptr) {
    Info(logger_, "adaptiveLinkedList Entry distribution among buckets: %s",
         keys_per_bucket_hist.ToString().c_str());
  }

  if (alloc_arena == nullptr) {
    return new FullListIterator(head, new_arena);
  } else {
    auto mem2 = alloc_arena->AllocateAligned(sizeof(FullListIterator));
    return new (mem2) FullListIterator(head, new_arena);
  }
}

MemTableRep::Iterator* AdaptiveLinkListRep::GetDynamicPrefixIterator(
    Arena* alloc_arena) {
  if (alloc_arena == nullptr) {
    return new DynamicIterator(*this);
  } else {
    auto mem = alloc_arena->AllocateAligned(sizeof(DynamicIterator));
    return new (mem) DynamicIterator(*this);
  }
}

bool AdaptiveLinkListRep::LinkListContains(Node* head,
                                       const Slice& user_key) const {
  Node* x = FindWetherInBucket(head, user_key);
  return (x != nullptr);
}



Node* AdaptiveLinkListRep::FindWetherInBucket(Node* head,
                                                  const Slice& key) const {
  Node* x = head;
  while (true) {
    if (x == nullptr) {
      return x;
    }
    // Make sure the lists are sorted.
    // If x points to head_ or next points nullptr, it is trivially satisfied.
    assert((x == head));
    if (compare_(x->key,key) == 0) {
      // Keep searching in this list
      return x;
    } 
    x = x->Next();
  }
  return x;
}

} // anon namespace

MemTableRep* AdaptiveLinkListRepFactory::CreateMemTableRep(
    const MemTableRep::KeyComparator& compare, Allocator* allocator,
    const SliceTransform* transform, Logger* logger) {
  return new AdaptiveLinkListRep(compare, allocator, transform, 
                             huge_page_tlb_size_,
                             logger, bucket_entries_logging_threshold_,
                             if_log_bucket_dist_when_flash_);
}

MemTableRepFactory* NewAdaptiveLinkListRepFactory(
    size_t huge_page_tlb_size,
    int bucket_entries_logging_threshold, bool if_log_bucket_dist_when_flash) {
  return new AdaptiveLinkListRepFactory(huge_page_tlb_size,
      bucket_entries_logging_threshold, if_log_bucket_dist_when_flash);
}

}  // namespace ROCKSDB_NAMESPACE

