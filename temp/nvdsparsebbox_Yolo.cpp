/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Edited by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "nvdsinfer_custom_impl.h"
#include <algorithm>
#include <iostream>
#include <vector>
#include <cmath>

extern "C" bool NvDsInferParseYolo(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

static float clamp(float val, float minVal, float maxVal)
{
    return std::max(minVal, std::min(val, maxVal));
}

extern "C" bool NvDsInferParseYolo(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    if (outputLayersInfo.empty()) return false;

    const NvDsInferLayerInfo& output = outputLayersInfo[0];
    const float* data = (const float*)output.buffer;

    // For YOLOv11 Standard Export:
    // output.inferDims.d[0] = 84 (attributes)
    // output.inferDims.d[1] = 8400 (anchors/boxes)
    const int numAttributes = output.inferDims.d[0]; 
    const int numAnchors = output.inferDims.d[1];
    const int numClasses = numAttributes - 4; // Should be 80

    for (int i = 0; i < numAnchors; i++)
    {
        // In transposed memory: 
        // X is at index i, Y at i + numAnchors, W at i + 2*numAnchors, etc.
        float bx = data[i];
        float by = data[i + numAnchors];
        float bw = data[i + 2 * numAnchors];
        float bh = data[i + 3 * numAnchors];

        float maxClassProb = 0.0f;
        int classId = -1;

        // Find the highest class probability for this anchor
        for (int c = 0; c < numClasses; c++)
        {
            // Class probabilities start after the 4 box coordinates
            float classProb = data[i + (4 + c) * numAnchors];
            if (classProb > maxClassProb)
            {
                maxClassProb = classProb;
                classId = c;
            }
        }

        if (classId < 0 || maxClassProb < detectionParams.perClassPreclusterThreshold[classId])
            continue;

        // YOLOv11 standard output is usually in absolute pixels based on training size (640)
        // If your input is different, you may need to scale these.
        float x1 = bx - bw / 2.0f;
        float y1 = by - bh / 2.0f;

        NvDsInferParseObjectInfo obj;
        obj.left = clamp(x1, 0.0f, (float)networkInfo.width);
        obj.top = clamp(y1, 0.0f, (float)networkInfo.height);
        obj.width = clamp(bw, 0.0f, (float)networkInfo.width);
        obj.height = clamp(bh, 0.0f, (float)networkInfo.height);
        obj.detectionConfidence = maxClassProb;
        obj.classId = classId;

        objectList.push_back(obj);
    }
    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo);
