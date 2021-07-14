// Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include <cstdio>
#include <string>
#include <unistd.h>
#include "rocksdb/db.h"
#include "rocksdb/slice.h"
#include "rocksdb/options.h"
#include "rocksdb/slice_transform.h"

using namespace ROCKSDB_NAMESPACE;

#if defined(OS_WIN)
std::string kDBPath = "C:\\Windows\\TEMP\\rocksdb_simple_example";
#else
std::string kDBPath = "/tmp/rocksdb_simple_example";
#endif

int main() {
  DB* db;
  Options options;
  options.memtable_factory.reset(ROCKSDB_NAMESPACE::NewHybridLinkListRepFactory(0,3,false));
  options.prefix_extractor.reset(NewFixedPrefixTransform(1));
  options.allow_concurrent_memtable_write = false;
  // Optimize RocksDB. This is the easiest way to get RocksDB to perform well
  // create the DB if it's not already present
  options.create_if_missing = true;

  // open DB
  Status s = DB::Open(options, kDBPath, &db);
  // Put key-value
  s = db->Put(WriteOptions(), "???", "1");
  std::string value;
  // get value
  s = db->Get(ReadOptions(), "???", &value);
  //assert(s.ok());
  assert(value == "1");
  /*
  s = db->Put(WriteOptions(), "22", "1");
  // get value
  s = db->Get(ReadOptions(), "22", &value);


  s = db->Put(WriteOptions(), "4444", "1");
  // get value
  s = db->Get(ReadOptions(), "44444", &value);
//
  s = db->Put(WriteOptions(), "55555", "555");
  std::string value_5;
  // get value
  s = db->Get(ReadOptions(), "55555", &value_5);

   s = db->Put(WriteOptions(), "6666666", "1");
  // get value
  s = db->Get(ReadOptions(), "6666666", &value);

 s = db->Put(WriteOptions(), "88888888", "1");
  // get value
  s = db->Get(ReadOptions(), "88888888", &value);

   s = db->Put(WriteOptions(), "1010101010", "1");
  // get value
  s = db->Get(ReadOptions(), "1010101010", &value);

  s = db->Put(WriteOptions(), "11111111111", "1");
  // get value
  s = db->Get(ReadOptions(), "11111111111", &value);

  */
  s = db->Put(WriteOptions(), "2", "1");
  // get value
  s = db->Get(ReadOptions(), "2", &value);


  s = db->Put(WriteOptions(), "5", "1");
  // get value
  s = db->Get(ReadOptions(), "5", &value);
//
  s = db->Put(WriteOptions(), "4", "555");
  std::string value_5;
  // get value
  s = db->Get(ReadOptions(), "4", &value_5);

   s = db->Put(WriteOptions(), "6", "1");
  // get value
  s = db->Get(ReadOptions(), "6", &value);

 s = db->Put(WriteOptions(), "8", "1");
  // get value
  s = db->Get(ReadOptions(), "8", &value);

   s = db->Put(WriteOptions(), "9", "1");
  // get value
  s = db->Get(ReadOptions(), "9", &value);

  FlushOptions no_wait;
  no_wait.wait = false;
  no_wait.allow_write_stall = true;
  db->Flush(no_wait);

  //sleep(1);
  //after first flush
  /*s = db->Put(WriteOptions(), "*9dag", "**");
  std::string value_9dag;
  // get value
  s = db->Get(ReadOptions(), "*9dag", &value_9dag);
  assert(value_9dag == "**");*/

  
  s = db->Put(WriteOptions(), "333", "1");
  // get value
  s = db->Get(ReadOptions(), "333", &value);

  s = db->Put(WriteOptions(), "7777777", "1111");
  std::string value_k1;
  // get value
  s = db->Get(ReadOptions(), "7777777", &value_k1);
  assert(value_k1 == "1111");

  s = db->Put(WriteOptions(), "999999999", "2222");
  std::string value_k2;
  // get value
  s = db->Get(ReadOptions(), "999999999", &value_k2);
  assert(value_k2 == "2222");

  db->Flush(no_wait);

  //sleep(1);
  //after second flush
  s = db->Put(WriteOptions(), "key3", "key3");
  std::string value_k3;
  // get value
  s = db->Get(ReadOptions(), "key3", &value_k3);
  assert(value_k3 == "key3");

  s = db->Put(WriteOptions(), "dafadf", "key4");
  std::string value_k4;
  // get value
  s = db->Get(ReadOptions(), "dafadf", &value_k4);
  assert(value_k4 == "key4");

  s = db->Put(WriteOptions(), "k", "key5");
  std::string value_k5;
  // get value
  s = db->Get(ReadOptions(), "k", &value_k5);
  assert(value_k5 == "key5");

  s = db->Put(WriteOptions(), "::::", "key6");
  std::string value_k6;
  // get value
  s = db->Get(ReadOptions(), "::::", &value_k6);
  assert(value_k6 == "key6");


  // atomically apply a set of updates

  /*
  {
    WriteBatch batch;
    batch.Put("key2", "world");
    s = db->Write(WriteOptions(), &batch);
  }

  s = db->Get(ReadOptions(), "key1", &value);
  assert(s.IsNotFound());

  db->Get(ReadOptions(), "key2", &value);
  //assert(value == "value");

  {
    PinnableSlice pinnable_val;
    db->Get(ReadOptions(), db->DefaultColumnFamily(), "key2", &pinnable_val);
    //assert(pinnable_val == "world");
  }

  {
    std::string string_val;
    // If it cannot pin the value, it copies the value to its internal buffer.
    // The intenral buffer could be set during construction.
    PinnableSlice pinnable_val(&string_val);
    db->Get(ReadOptions(), db->DefaultColumnFamily(), "key2", &pinnable_val);
    //assert(pinnable_val == "world");
    // If the value is not pinned, the internal buffer must have the value.
    //assert(pinnable_val.IsPinned() || string_val == "world");
  }

  PinnableSlice pinnable_val;
  //s = db->Get(ReadOptions(), db->DefaultColumnFamily(), "key1", &pinnable_val);
  //assert(s.IsNotFound());
  // Reset PinnableSlice after each use and before each reuse
  pinnable_val.Reset();
  db->Get(ReadOptions(), db->DefaultColumnFamily(), "key2", &pinnable_val);
  //assert(pinnable_val == "world");
  pinnable_val.Reset();
  // The Slice pointed by pinnable_val is not valid after this point
  */
  delete db;

  return 0;
}

