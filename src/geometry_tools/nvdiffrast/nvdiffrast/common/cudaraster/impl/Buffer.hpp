// Copyright (c) 2009-2022, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once
#include "Defs.hpp"

namespace CR
{
//------------------------------------------------------------------------

class Buffer
{
public:
                    Buffer      (void);
                    ~Buffer     (void);

    void            reset       (size_t bytes);
    void            grow        (size_t bytes);
    void*           getPtr      (void) { return m_gpuPtr; }
    size_t          getSize     (void) const { return m_bytes; }

    void            setPtr      (void* ptr) { m_gpuPtr = ptr; }

private:
    void*           m_gpuPtr;
    size_t          m_bytes;
};

//------------------------------------------------------------------------
}
