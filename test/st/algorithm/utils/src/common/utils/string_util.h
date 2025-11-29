/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STRING_UTIL_H
#define STRING_UTIL_H

#include <string>
#include <vector>
#include <sstream>
#include "securec.h"

template <typename... Args> inline std::string StringFormat(const char *format, Args... args)
{
    using namespace std;
    constexpr size_t bufSize = BUFSIZ;
    char             buffer[bufSize];
    size_t           actualSize = snprintf_s(&buffer[0], bufSize, bufSize, format, args...);
    actualSize++;

    if (actualSize > bufSize) {
        std::vector<char> newbuffer(actualSize);
        snprintf_s(newbuffer.data(), actualSize, actualSize, format, args...);
        return newbuffer.data();
    }
    return buffer;
}

template <typename I> std::string Dec2hex(I i)
{
    static_assert(std::is_integral<I>::value, "type I is not a integral");
    std::stringstream ss;
    ss << std::hex << "0x" << i;
    return ss.str();
}

std::vector<std::string> SplitString(const std::string &str, const char c);

template <class T> T String2T(const std::string &s)
{
    // T must support >>
    T                  t;
    std::istringstream ist(s);
    ist >> t;
    return t;
}

#endif