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
std::string kDBPath = "/tmp/rocksdb_simple_example2";
#endif

int main() {
  DB* db;
  Options options;
  options.memtable_factory.reset(ROCKSDB_NAMESPACE::NewHybridLinkListRepFactory(0,1,false));
  options.prefix_extractor.reset(NewFixedPrefixTransform(1));
  options.allow_concurrent_memtable_write = false;
  // Optimize RocksDB. This is the easiest way to get RocksDB to perform well
  options.IncreaseParallelism();
  options.OptimizeLevelStyleCompaction();
  // create the DB if it's not already present
  options.create_if_missing = true;

  // open DB
  Status s = DB::Open(options, kDBPath, &db);

  s = db->Put(WriteOptions(), "1", "4");
  std::string value;
  s = db->Get(ReadOptions(), "1", &value);
  assert(value == "4");

  s = db->Put(WriteOptions(), "5", "6");
  std::string value_5;
  s = db->Get(ReadOptions(), "5", &value_5);
  assert(value_5 == "6");

  s = db->Put(WriteOptions(), "2", "9");
  std::string value_2;
  s = db->Get(ReadOptions(), "2", &value_2);
  assert(value_2 == "9");

  s = db->Put(WriteOptions(), "3", "3");
  std::string value_3;
  s = db->Get(ReadOptions(), "3", &value_3);
  assert(value_3 == "3");

  s = db->Put(WriteOptions(), "2", "5");
  s = db->Get(ReadOptions(), "2", &value_2);
  assert(value_2 == "5");

  s = db->Put(WriteOptions(), "99", "9");
  std::string value_99;
  s = db->Get(ReadOptions(), "99", &value_99);
  assert(value_99 == "9");

  FlushOptions no_wait;
  db->Flush(no_wait);

  s = db->Put(WriteOptions(), "01", "4");
  s = db->Get(ReadOptions(), "01", &value);
  assert(value == "4");

  s = db->Put(WriteOptions(), "59", "6");
  s = db->Get(ReadOptions(), "59", &value_5);
  assert(value_5 == "6");

  s = db->Put(WriteOptions(), "72", "9");
  s = db->Get(ReadOptions(), "72", &value_2);
  assert(value_2 == "9");

  s = db->Put(WriteOptions(), "83", "3");
  s = db->Get(ReadOptions(), "83", &value_3);
  assert(value_3 == "3");

  s = db->Put(WriteOptions(), "2", "5");
  s = db->Get(ReadOptions(), "2", &value_2);
  assert(value_2 == "5");

  s = db->Put(WriteOptions(), "99", "9");
  s = db->Get(ReadOptions(), "99", &value_99);
  assert(value_99 == "9");

  db->Flush(no_wait);

  s = db->Put(WriteOptions(), "8", "4");
  s = db->Get(ReadOptions(), "8", &value);
  assert(value == "4");

  s = db->Put(WriteOptions(), "21", "1");
  s = db->Get(ReadOptions(), "21", &value);
  assert(value == "1");

  s = db->Put(WriteOptions(), "89", "9");
  s = db->Get(ReadOptions(), "89", &value);
  assert(value == "9");

  s = db->Put(WriteOptions(), "04", "3");
  s = db->Get(ReadOptions(), "04", &value);
  assert(value == "3");

  s = db->Put(WriteOptions(), "2", "5");
  s = db->Get(ReadOptions(), "2", &value_2);
  assert(value_2 == "5");

  s = db->Put(WriteOptions(), "11999", "9");
  s = db->Get(ReadOptions(), "11999", &value);
  assert(value == "9");

  db->Flush(no_wait);

  delete db;

  return 0;
}