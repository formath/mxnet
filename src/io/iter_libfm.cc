/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file iter_libfm.cc
 * \brief define a LibFM Reader to read in arrays
 */
#include <mxnet/io.h>
#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <dmlc/data.h>
#include "./iter_sparse_prefetcher.h"
#include "./iter_sparse_batchloader.h"

namespace mxnet {
namespace io {
// LibSVM parameters
struct LibFMIterParam : public dmlc::Parameter<LibFMIterParam> {
  /*! \brief path to data libfm file */
  std::string data_libfm;
  /*! \brief data shape */
  TShape data_shape;
  /*! \brief path to label libfm file */
  std::string label_libfm;
  /*! \brief label shape */
  TShape label_shape;
  /*! \brief partition the data into multiple parts */
  int num_parts;
  /*! \brief the index of the part will read*/
  int part_index;
  // declare parameters
  DMLC_DECLARE_PARAMETER(LibFMIterParam) {
    DMLC_DECLARE_FIELD(data_libfm)
        .describe("The input LibFM file or a directory path.");
    DMLC_DECLARE_FIELD(data_shape)
        .describe("The shape of one example.");
    DMLC_DECLARE_FIELD(label_libfm).set_default("NULL")
        .describe("The input LibFM file or a directory path. "
                  "If NULL, all labels will be read from ``data_libfm``.");
    index_t shape1[] = {1};
    DMLC_DECLARE_FIELD(label_shape).set_default(TShape(shape1, shape1 + 1))
        .describe("The shape of one label.");
    DMLC_DECLARE_FIELD(num_parts).set_default(1)
        .describe("partition the data into multiple parts");
    DMLC_DECLARE_FIELD(part_index).set_default(0)
        .describe("the index of the part will read");
  }
};

class LibFMIter: public SparseIIterator<DataInst> {
 public:
  LibFMIter() {}
  virtual ~LibFMIter() {}

  // intialize iterator loads data in
  virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    param_.InitAllowUnknown(kwargs);
    CHECK_EQ(param_.data_shape.ndim(), 1) << "dimension of data_shape is expected to be 1";
    CHECK_GT(param_.num_parts, 0) << "number of parts should be positive";
    CHECK_GE(param_.part_index, 0) << "part index should be non-negative";
    data_parser_.reset(dmlc::Parser<uint64_t>::Create(param_.data_libfm.c_str(),
                                                      param_.part_index,
                                                      param_.num_parts, "libfm"));
    if (param_.label_libfm != "NULL") {
      label_parser_.reset(dmlc::Parser<uint64_t>::Create(param_.label_libfm.c_str(),
                                                         param_.part_index,
                                                         param_.num_parts, "libfm"));
      CHECK_GT(param_.label_shape.Size(), 1)
        << "label_shape is not expected to be (1,) when param_.label_libfm is set.";
    } else {
      CHECK_EQ(param_.label_shape.Size(), 1)
        << "label_shape is expected to be (1,) when param_.label_libfm is NULL";
    }
    // both data and label are of CSRStorage in libfm format
    if (param_.label_shape.Size() > 1) {
      out_.data.resize(8);
    } else {
      // only data is of CSRStorage in libfm format.
      out_.data.resize(5);
    }
  }

  virtual void BeforeFirst() {
    data_parser_->BeforeFirst();
    if (label_parser_.get() != nullptr) {
      label_parser_->BeforeFirst();
    }
    data_ptr_ = label_ptr_ = 0;
    data_size_ = label_size_ = 0;
    inst_counter_ = 0;
    end_ = false;
  }

  virtual bool Next() {
    if (end_) return false;
    while (data_ptr_ >= data_size_) {
      if (!data_parser_->Next()) {
        end_ = true; return false;
      }
      data_ptr_ = 0;
      data_size_ = data_parser_->Value().size;
    }
    out_.index = inst_counter_++;
    CHECK_LT(data_ptr_, data_size_);
    const auto data_row = data_parser_->Value()[data_ptr_++];
    // data, fieldes, indices and indptr
    out_.data[0] = AsDataBlob(data_row);
    out_.data[1] = AsFieldBlob(data_row);
    out_.data[2] = AsIdxBlob(data_row);
    out_.data[3] = AsIndPtrPlaceholder(data_row);

    if (label_parser_.get() != nullptr) {
      while (label_ptr_ >= label_size_) {
        CHECK(label_parser_->Next())
            << "Data LibFM's row is smaller than the number of rows in label_libfm";
        label_ptr_ = 0;
        label_size_ = label_parser_->Value().size;
      }
      CHECK_LT(label_ptr_, label_size_);
      const auto label_row = label_parser_->Value()[label_ptr_++];
      // data, fieldes, indices and indptr
      out_.data[4] = AsDataBlob(label_row);
      out_.data[5] = AsFieldBlob(label_row);
      out_.data[6] = AsIdxBlob(label_row);
      out_.data[7] = AsIndPtrPlaceholder(label_row);
    } else {
      out_.data[4] = AsScalarLabelBlob(data_row);
    }
    return true;
  }

  virtual const DataInst &Value(void) const {
    return out_;
  }

  virtual const NDArrayStorageType GetStorageType(bool is_data) const {
    if (is_data) return kCSRStorage;
    return param_.label_shape.Size() > 1 ? kCSRStorage : kDefaultStorage;
  }

  virtual const TShape GetShape(bool is_data) const {
    if (is_data) return param_.data_shape;
    return param_.label_shape;
  }

 private:
  inline TBlob AsDataBlob(const dmlc::Row<uint64_t>& row) {
    const real_t* ptr = row.value;
    TShape shape(mshadow::Shape1(row.length));
    return TBlob((real_t*) ptr, shape, cpu::kDevMask);  // NOLINT(*)
  }

  inline TBlob AsFieldBlob(const dmlc::Row<uint64_t>& row) {
    const uint64_t* ptr = row.field;
    TShape shape(mshadow::Shape1(row.length));
    return TBlob((int64_t*) ptr, shape, cpu::kDevMask, mshadow::kInt64);  // NOLINT(*)
  }

  inline TBlob AsIdxBlob(const dmlc::Row<uint64_t>& row) {
    const uint64_t* ptr = row.index;
    TShape shape(mshadow::Shape1(row.length));
    return TBlob((int64_t*) ptr, shape, cpu::kDevMask, mshadow::kInt64);  // NOLINT(*)
  }

  inline TBlob AsIndPtrPlaceholder(const dmlc::Row<uint64_t>& row) {
    return TBlob(nullptr, mshadow::Shape1(0), cpu::kDevMask, mshadow::kInt64);
  }

  inline TBlob AsScalarLabelBlob(const dmlc::Row<uint64_t>& row) {
    const real_t* ptr = row.label;
    return TBlob((real_t*) ptr, mshadow::Shape1(1), cpu::kDevMask);  // NOLINT(*)
  }

  LibFMIterParam param_;
  // output instance
  DataInst out_;
  // internal instance counter
  unsigned inst_counter_{0};
  // at end
  bool end_{false};
  // label parser
  size_t label_ptr_{0}, label_size_{0};
  size_t data_ptr_{0}, data_size_{0};
  std::unique_ptr<dmlc::Parser<uint64_t> > label_parser_;
  std::unique_ptr<dmlc::Parser<uint64_t> > data_parser_;
};


DMLC_REGISTER_PARAMETER(LibFMIterParam);

MXNET_REGISTER_IO_ITER(LibFMIter)
.describe(R"code(Returns the libfm file iterator which returns sparse data with `csr`
storage type. This iterator is experimental and should be used with care.

The input data is stored in a format similar to libsvm file format, except that the libfm has a field. Namely, libsvm looks like `featureid:value` while libfm like `fieldid:featureid:value`.

In this function, the `data_shape` parameter is used to set the shape of each line of the data.
The dimension of both `data_shape` and `label_shape` are expected to be 1.

When `label_libfm` is set to ``NULL``, both data and label are read from the same file specified
by `data_libfm`. In this case, the data is stored in `csr` storage type, while the label is a 1D
dense array. Otherwise, data is read from `data_libfm` and label from `label_libfm`,
in this case, both data and label are stored in csr storage type. If `data_libfm` contains label,
it will ignored.

The `LibFMIter` only support `round_batch` parameter set to ``True`` for now. So, if `batch_size`
is 3 and there are 4 total rows in libfm file, 2 more examples
are consumed at the first round. If `reset` function is called after first round,
the call is ignored and remaining examples are returned in the second round.

If ``data_libfm = 'data/'`` is set, then all the files in this directory will be read.

Examples::

  # Contents of libfm file ``data.t``.
  1.0 0:0:0.5 1:2:1.2
  -2.0
  -3.0 0:4:0.6 1:8:2.4 2:10:1.2
  4 2:738:-1.2

  # Creates a `LibFMIter` with `batch_size`=3.
  >>> data_iter = mx.io.LibFMIter(data_libfm = 'data.t', data_shape = (3,), batch_size = 3)
  # The data of the first batch is stored in csr storage type
  >>> batch = data_iter.next()
  >>> csr = batch.data[0]
  <CSRNDArray 3x3 @cpu(0)>
  >>> csr.asnumpy()
  [[ 0.5        0.          1.2 ]
  [ 0.          0.          0.  ]
  [ 0.6         2.4         1.2]]
  # The label of first batch
  >>> label = batch.label[0]
  >>> label
  [ 1. -2. -3.]
  <NDArray 3 @cpu(0)>

  >>> second_batch = data_iter.next()
  # The data of the second batch
  >>> second_batch.data[0].asnumpy()
  [[ 0.          0.         -1.2 ]
   [ 0.5         0.          1.2 ]
   [ 0.          0.          0. ]]
  # The label of the second batch
  >>> second_batch.label[0].asnumpy()
  [ 4.  1. -2.]

  # Contents of libfm file ``label.t``
  1.0
  -2.0 0:0.125
  -3.0 2:1.2
  4 1:1.0 2:-1.2

  # Creates a `LibFMIter` with specified label file
  >>> data_iter = mx.io.LibFMIter(data_libfm = 'data.t', data_shape = (3,),
                   label_libfm = 'label.t', label_shape = (3,), batch_size = 3)

  # Both data and label are in csr storage type
  >>> batch = data_iter.next()
  >>> csr_data = batch.data[0]
  <CSRNDArray 3x3 @cpu(0)>
  >>> csr_data.asnumpy()
  [[ 0.5         0.          1.2  ]
   [ 0.          0.          0.   ]
   [ 0.6         2.4         1.2 ]]
  >>> csr_label = batch.label[0]
  <CSRNDArray 3x3 @cpu(0)>
  >>> csr_label.asnumpy()
  [[ 0.          0.          0.   ]
   [ 0.125       0.          0.   ]
   [ 0.          0.          1.2 ]]

)code" ADD_FILELINE)
.add_arguments(LibFMIterParam::__FIELDS__())
.add_arguments(BatchParam::__FIELDS__())
.add_arguments(PrefetcherParam::__FIELDS__())
.set_body([]() {
    return new SparsePrefetcherIter(
        new SparseBatchLoader(
            new LibFMIter()));
  });

}  // namespace io
}  // namespace mxnet