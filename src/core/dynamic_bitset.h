#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <limits>
#include <cassert>

// This is implented directly in the header, beacase this allows the library to stay header-only
class DynamicBitset {
public:
    inline DynamicBitset(size_t size) : size_(size), data_((size + 63) / 64) {
    }

    inline void reset() {
        for(uint64_t& x : data_)
            x = 0;
    }

    struct Accessor {
        friend class DynamicBitset;

        inline operator bool() const {
            return ((bitset_->data_[index_ / 64] >> (index_ % 64)) & 1) > 0;
        }

        inline Accessor& operator=(bool val) {
            if(val)
                bitset_->data_[index_ / 64] |= static_cast<uint64_t>(1) << (index_ % 64);
            else
                bitset_->data_[index_ / 64] &= std::numeric_limits<uint64_t>::max() - (static_cast<uint64_t>(1) << (index_ % 64));
            return *this;
        }

    private:
        inline Accessor(DynamicBitset* bitset, size_t index) {
            bitset_ = bitset;
            index_ = index;
        }

        DynamicBitset* bitset_;
        size_t index_;
    };

    struct ConstAccessor {
        friend class DynamicBitset;

        inline operator bool() const {
            return ((bitset_->data_[index_ / 64] >> (index_ % 64)) & 1) > 0;
        }

    private:
        inline ConstAccessor(const DynamicBitset* bitset, size_t index) {
            bitset_ = bitset;
            index_ = index;
        }

        const DynamicBitset* bitset_;
        size_t index_;
    };

    inline Accessor operator[](size_t ind) {
        return {this, ind};
    }

    inline ConstAccessor operator[](size_t ind) const {
        return {this, ind};
    }

    inline DynamicBitset operator|(const DynamicBitset& other) const {
        assert(size_ == other.size_);

        DynamicBitset res(size_);
        for(size_t i = 0; i < data_.size(); i++)
            res.data_[i] = data_[i] | other.data_[i];

        return res;
    }

    inline DynamicBitset& operator|=(const DynamicBitset& other) {
        assert(size_ == other.size_);

        for(size_t i = 0; i < data_.size(); i++)
            data_[i] |= other.data_[i];

        return *this;
    }

private:
    size_t size_;
    std::vector<uint64_t> data_;
};
