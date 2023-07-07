// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include <opencv2/imgproc.hpp>

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

Image2BlobParams::Image2BlobParams():scalefactor(Scalar::all(1.0)), size(Size()), mean(Scalar()), swapRB(false), ddepth(CV_32F),
                           datalayout(DNN_LAYOUT_NCHW), paddingmode(DNN_PMODE_NULL)
{}

Image2BlobParams::Image2BlobParams(const Scalar& scalefactor_, const Size& size_, const Scalar& mean_, bool swapRB_,
                         int ddepth_, DataLayout datalayout_, ImagePaddingMode mode_):
        scalefactor(scalefactor_), size(size_), mean(mean_), swapRB(swapRB_), ddepth(ddepth_),
        datalayout(datalayout_), paddingmode(mode_)
{}

void getVector(InputArrayOfArrays images_, std::vector<Mat>& images) {
    images_.getMatVector(images);
}

void getVector(InputArrayOfArrays images_, std::vector<UMat>& images) {
    images_.getUMatVector(images);
}

void getMat(UMat& blob, InputArray blob_, AccessFlag flag) {
    if(blob_.kind() == _InputArray::UMAT)
        blob = blob_.getUMat();
    else if(blob_.kind() == _InputArray::MAT) {
        blob = blob_.getMat().getUMat(flag);
    }
}

void getMat(Mat& blob, InputArray blob_, AccessFlag flag) {
    if(blob_.kind() == _InputArray::UMAT)
        blob = blob_.getUMat().getMat(flag);
    else if(blob_.kind() == _InputArray::MAT) {
        blob = blob_.getMat();
    }
}

void makeMatFromBlob(Mat& m, InputArray blob, int i, int j, int rows, int cols, int type) {
    m = Mat(rows, cols, type, blob.getMat().ptr(i, j));
}

void makeMatFromBlob(UMat& m, InputArray blob, int i, int j, int rows, int cols, int type) {
    UMat ublob = blob.getUMat();
    int offset = i * cols + j;
    int length = 1;
    for(int i = 0; i < ublob.dims; ++i) {
        length *= ublob.size[i];
    }
    const int newShape[1] { length };
    UMat reshaped;
    reshaped = ublob.reshape(1, 1, newShape);
    UMat sub = reshaped(Rect(0, offset, 1, rows * cols));
    m = sub.reshape(CV_MAT_CN(type), rows);
    assert(m.type() == type);
}


void blobFromImage(InputArray image, OutputArray blob, double scalefactor,
        const Size& size, const Scalar& mean, bool swapRB, bool crop, int ddepth)
{
    CV_TRACE_FUNCTION();
    if(image.kind() == _InputArray::UMAT) {
        UMat u = image.getUMat();
        std::vector<UMat> vec(1, u);
        blobFromImages(vec, blob, scalefactor, size, mean, swapRB, crop, ddepth);
    } else if(image.kind() == _InputArray::MAT) {
        Mat m = image.getMat();
        std::vector<Mat> vec(1, m);
        blobFromImages(vec, blob, scalefactor, size, mean, swapRB, crop, ddepth);
    } else {
        assert(false);
    }
}

Mat blobFromImage(InputArray image, const double scalefactor, const Size& size,
        const Scalar& mean, bool swapRB, bool crop, int ddepth)
{
    CV_TRACE_FUNCTION();
    Mat blob;
    if(image.kind() == _InputArray::UMAT) {
        UMat ublob = blob.getUMat(cv::ACCESS_RW);
        blobFromImage(image, blob, scalefactor, size, mean, swapRB, crop, ddepth);
    } else if(image.kind() == _InputArray::MAT) {
        blobFromImage(image, blob, scalefactor, size, mean, swapRB, crop, ddepth);
    }

    return blob;
}

Mat blobFromImages(InputArrayOfArrays images, double scalefactor, Size size,
        const Scalar& mean, bool swapRB, bool crop, int ddepth)
{
    CV_TRACE_FUNCTION();
    UMat blob;
    blobFromImages(images, blob, scalefactor, size, mean, swapRB, crop, ddepth);
    return blob.getMat(ACCESS_RW).clone();
}

template<class Tmat>
void blobFromImagesWithParams(InputArrayOfArrays images_, Tmat& blob_, const Image2BlobParams& param)
{
    CV_TRACE_FUNCTION();
    bool isUMat = std::is_same<Tmat, UMat>::value;
    if(!isUMat && !std::is_same<Tmat, Mat>::value) {
        String error_message = "The template parameter is expected to be either a cv::Mat or a cv::UMat";
        CV_Error(Error::StsBadArg, error_message);
    }

    CV_CheckType(param.ddepth, param.ddepth == CV_32F || param.ddepth == CV_8U,
                 "Blob depth should be CV_32F or CV_8U");
    Size size = param.size;

    std::vector<Tmat> images;
    getVector(images_, images);

    CV_Assert(!images.empty());

    int nch = images[0].channels();
    Scalar scalefactor = param.scalefactor;

    if (param.ddepth == CV_8U)
    {
        CV_Assert(scalefactor == Scalar::all(1.0) && "Scaling is not supported for CV_8U blob depth");
        CV_Assert(param.mean == Scalar() && "Mean subtraction is not supported for CV_8U blob depth");
    }

    for (size_t i = 0; i < images.size(); i++)
    {
        Size imgSize = images[i].size();
        if (size == Size())
            size = imgSize;
        if (size != imgSize)
        {
            if (param.paddingmode == DNN_PMODE_CROP_CENTER)
            {
                float resizeFactor = std::max(size.width / (float)imgSize.width,
                                              size.height / (float)imgSize.height);
                resize(images[i], images[i], Size(), resizeFactor, resizeFactor, INTER_LINEAR);
                Rect crop(Point(0.5 * (images[i].cols - size.width),
                                0.5 * (images[i].rows - size.height)),
                          size);
                images[i] = images[i](crop);
            }
            else
            {
                if (param.paddingmode == DNN_PMODE_LETTERBOX)
                {
                    float resizeFactor = std::min(size.width / (float)imgSize.width,
                                                  size.height / (float)imgSize.height);
                    int rh = int(imgSize.height * resizeFactor);
                    int rw = int(imgSize.width * resizeFactor);
                    resize(images[i], images[i], Size(rw, rh), INTER_LINEAR);

                    int top = (size.height - rh)/2;
                    int bottom = size.height - top - rh;
                    int left = (size.width - rw)/2;
                    int right = size.width - left - rw;
                    copyMakeBorder(images[i], images[i], top, bottom, left, right, BORDER_CONSTANT);
                }
                else
                    resize(images[i], images[i], size, 0, 0, INTER_LINEAR);
            }
        }

        Scalar mean = param.mean;
        if (param.swapRB)
        {
            std::swap(mean[0], mean[2]);
            std::swap(scalefactor[0], scalefactor[2]);
        }

        if (images[i].depth() == CV_8U && param.ddepth == CV_32F)
            images[i].convertTo(images[i], CV_32F);

        subtract(images[i], mean, images[i]);
        multiply(images[i], scalefactor, images[i]);
    }

    size_t nimages = images.size();
    Tmat image0 = images[0];
    CV_Assert(image0.dims == 2);

    if (param.datalayout == DNN_LAYOUT_NCHW)
    {
        if (nch == 3 || nch == 4)
        {
            int sz[] = { (int)nimages, nch, image0.rows, image0.cols };
            blob_.create(4, sz, param.ddepth);
            int szr[1] = { int(nimages * nch * image0.rows * image0.cols) };
            blob_.reshape(1, 1, szr) = cv::Scalar::all(0);
            std::vector<Tmat> ch(4);
            ch[0] = Tmat(image0.size(), CV_MAT_DEPTH(image0.type()));
            ch[1] = Tmat(image0.size(), CV_MAT_DEPTH(image0.type()));
            ch[2] = Tmat(image0.size(), CV_MAT_DEPTH(image0.type()));
            ch[3] = Tmat(image0.size(), CV_MAT_DEPTH(image0.type()));
            ch[0] = cv::Scalar::all(0);
            ch[1] = cv::Scalar::all(0);
            ch[2] = cv::Scalar::all(0);
            ch[3] = cv::Scalar::all(0);

            for (size_t i = 0; i < nimages; i++)
            {
                const Tmat& image = images[i];
                CV_Assert(image.depth() == blob_.depth());
                nch = image.channels();
                CV_Assert(image.dims == 2 && (nch == 3 || nch == 4));
                CV_Assert(image.size() == image0.size());

                for (int j = 0; j < nch; j++) {
                    makeMatFromBlob(ch[j], blob_, i, j ,image.rows, image.cols, param.ddepth);
                }
                if (param.swapRB)
                    std::swap(ch[0], ch[2]);

                split(image, ch);
            }
        }
        else
        {
            CV_Assert(nch == 1);
            int sz[] = { (int)nimages, 1, image0.rows, image0.cols };
            blob_.create(4, sz, param.ddepth);
            int szr[1] = { int(nimages * 1 * image0.rows * image0.cols) };
            blob_.reshape(1, 1, szr) = cv::Scalar::all(0);
            Mat blob;
            getMat(blob, blob_, ACCESS_RW);

            for (size_t i = 0; i < nimages; i++)
            {
                const Tmat& image = images[i];
                CV_Assert(image.depth() == blob_.depth());
                nch = image.channels();
                CV_Assert(image.dims == 2 && (nch == 1));
                CV_Assert(image.size() == image0.size());

                image.copyTo(Mat(image.rows, image.cols, param.ddepth, blob.ptr((int)i, 0)));
            }
        }
    }
    else if (param.datalayout == DNN_LAYOUT_NHWC)
    {
        int sz[] = { (int)nimages, image0.rows, image0.cols, nch};
        blob_.create(4, sz, param.ddepth);
        int szr[1] = { int(nimages * nch * image0.rows * image0.cols) };
        blob_.reshape(1, 1, szr) = cv::Scalar::all(0);
        Mat blob;
        getMat(blob, blob_, ACCESS_RW);
        int subMatType = CV_MAKETYPE(param.ddepth, nch);
        for (size_t i = 0; i < nimages; i++)
        {
            const Tmat& image = images[i];
            CV_Assert(image.depth() == blob_.depth());
            CV_Assert(image.channels() == image0.channels());
            CV_Assert(image.size() == image0.size());
            if (param.swapRB)
            {
                Mat tmpRB;
                cvtColor(image, tmpRB, COLOR_BGR2RGB);
                tmpRB.copyTo(Mat(tmpRB.rows, tmpRB.cols, subMatType, blob.ptr((int)i, 0)));
            }
            else
                image.copyTo(Mat(image.rows, image.cols, subMatType, blob.ptr((int)i, 0)));
        }
    }
    else
        CV_Error(Error::StsUnsupportedFormat, "Unsupported data layout in blobFromImagesWithParams function.");
}

void blobFromImages(InputArrayOfArrays images_, OutputArray blob_, double scalefactor,
        Size size, const Scalar& mean_, bool swapRB, bool crop, int ddepth)
{
    CV_TRACE_FUNCTION();
    if (images_.kind() != _InputArray::STD_VECTOR_UMAT &&
        images_.kind() != _InputArray::STD_VECTOR_MAT && images_.kind() != _InputArray::STD_ARRAY_MAT &&
        images_.kind() != _InputArray::STD_VECTOR_VECTOR) {
        String error_message = "The data is expected as vectors of vectors or vectors of matrices.";
        CV_Error(Error::StsBadArg, error_message);
    }
    Image2BlobParams param(Scalar::all(scalefactor), size, mean_, swapRB, ddepth);
    if (crop)
        param.paddingmode = DNN_PMODE_CROP_CENTER;
    if(blob_.kind() == _InputArray::UMAT) {
        UMat u;
        blobFromImagesWithParams<UMat>(images_, u, param);
        u.copyTo(blob_);
    } else if(blob_.kind() == _InputArray::MAT) {
        Mat m;
        blobFromImagesWithParams<Mat>(images_, m, param);
        m.copyTo(blob_);
    }
    else
        assert(false);
}

Mat blobFromImageWithParams(InputArray image, const Image2BlobParams& param)
{
    CV_TRACE_FUNCTION();
    if(image.kind() == _InputArray::UMAT) {
        cv::UMat blob;
        std::vector<UMat> images(1, image.getUMat());
        blobFromImagesWithParams<cv::UMat>(images, blob, param);
        return blob.getMat(ACCESS_RW).clone();
    } else if(image.kind() == _InputArray::MAT) {
        cv::Mat blob;
        std::vector<Mat> images(1, image.getMat());
        blobFromImagesWithParams<cv::Mat>(images, blob, param);
        return blob;
    } else
        assert(false); //FIXME

    return {};
}

void blobFromImageWithParams(InputArray image, OutputArray blob, const Image2BlobParams& param)
{
    CV_TRACE_FUNCTION();
    if(image.kind() == _InputArray::UMAT && blob.kind() == _InputArray::UMAT) {
        std::vector<UMat> images(1, image.getUMat());
        UMat u = blob.getUMat();
        blobFromImagesWithParams<cv::UMat>(images, u, param);
    } else if(image.kind() == _InputArray::MAT && blob.kind() == _InputArray::MAT) {
        std::vector<Mat> images(1, image.getMat());
        Mat m = blob.getMat();
        blobFromImagesWithParams<cv::Mat>(images, m, param);

    } else
        assert(false); //FIXME
}

Mat blobFromImagesWithParams(InputArrayOfArrays images, const Image2BlobParams& param)
{
    CV_TRACE_FUNCTION();
    if(images.kind() == _InputArray::STD_VECTOR_UMAT) {
        UMat blob;
        blobFromImagesWithParams<UMat>(images, blob, param);
        return blob.getMat(ACCESS_RW).clone();
    } else {
        Mat blob;
        blobFromImagesWithParams<Mat>(images, blob, param);
        return blob;
    }
}

void blobFromImagesWithParams(InputArrayOfArrays images_, OutputArray blob_, const Image2BlobParams& param)
{
    CV_TRACE_FUNCTION();
    if (images_.kind() != _InputArray::STD_VECTOR_MAT && images_.kind() != _InputArray::STD_VECTOR_UMAT &&
            images_.kind() != _InputArray::STD_ARRAY_MAT && images_.kind() != _InputArray::STD_VECTOR_VECTOR) {
        String error_message = "The data is expected as vectors of vectors or vectors of (u)matrices.";
        CV_Error(Error::StsBadArg, error_message);
    }
    CV_CheckType(param.ddepth, param.ddepth == CV_32F || param.ddepth == CV_8U,
                 "Blob depth should be CV_32F or CV_8U");

    if(images_.kind() == _InputArray::STD_VECTOR_MAT) {
        if(blob_.kind() == _InputArray::MAT) {
            Mat m = blob_.getMat();
            blobFromImagesWithParams<Mat>(images_, m, param);
        } else if(blob_.kind() == _InputArray::UMAT) {
            Mat m = blob_.getUMat().getMat(ACCESS_RW);
            blobFromImagesWithParams<Mat>(images_, m, param);
        }
    }
    else if(images_.kind() == _InputArray::STD_VECTOR_UMAT) {
        if(blob_.kind() == _InputArray::MAT) {
            UMat u = blob_.getMat().getUMat(ACCESS_RW);
            blobFromImagesWithParams<UMat>(images_, u, param);
        } else if(blob_.kind() == _InputArray::UMAT) {
            UMat u = blob_.getUMat();
            blobFromImagesWithParams<UMat>(images_, u, param);
        }

    } else
        assert(false); //FIXME

}

void imagesFromBlob(const cv::Mat& blob_, OutputArrayOfArrays images_)
{
    CV_TRACE_FUNCTION();

    // A blob is a 4 dimensional matrix in floating point precision
    // blob_[0] = batchSize = nbOfImages
    // blob_[1] = nbOfChannels
    // blob_[2] = height
    // blob_[3] = width
    CV_Assert(blob_.depth() == CV_32F);
    CV_Assert(blob_.dims == 4);

    images_.create(cv::Size(1, blob_.size[0]), blob_.depth());

    std::vector<Mat> vectorOfChannels(blob_.size[1]);
    for (int n = 0; n < blob_.size[0]; ++n)
    {
        for (int c = 0; c < blob_.size[1]; ++c)
        {
            vectorOfChannels[c] = getPlane(blob_, n, c);
        }
        cv::merge(vectorOfChannels, images_.getMatRef(n));
    }
}


CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
