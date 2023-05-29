#pragma once

#include "types.h"

namespace vnr {

template<typename T>
struct FrameBufferTemplate
{
private:
  struct BufferObject
  {
    cudaStream_t stream{};
    CUDABuffer device_buffer;
    std::vector<T> host_buffer;

    ~BufferObject()
    {
      device_buffer.free(stream);
    }

    void resize(size_t& count)
    {
      device_buffer.resize(count * sizeof(T), stream);
      host_buffer.resize(count);
    }

    void reset()
    {
      CUDA_CHECK(cudaMemsetAsync((void*)device_buffer.d_pointer(), 0, device_buffer.sizeInBytes, stream));
    }

    void create() { CUDA_CHECK(cudaStreamCreate(&stream)); }

    void download_async() { device_buffer.download_async(host_buffer.data(), host_buffer.size(), stream); }

    T* d_pointer() const { return (T*)device_buffer.d_pointer(); }

    T* h_pointer() const { return (T*)host_buffer.data(); }

    void deepcopy(void* dst) { std::memcpy(dst, host_buffer.data(), host_buffer.size() * sizeof(T)); }
  };

  BufferObject& current_buffer() { return buffers[current_buffer_index]; }

  const BufferObject& current_buffer() const { return buffers[current_buffer_index]; }

  BufferObject buffers[2];
  int current_buffer_index{ 0 };

  size_t fb_pixel_count{ 0 };
  vec2i fb_size;

public:
  ~FrameBufferTemplate() {}

  void create()
  {
    buffers[0].create();
    buffers[1].create();
  }

  void resize(vec2i s)
  {
    fb_size = s;
    fb_pixel_count = (size_t)fb_size.x * fb_size.y;
    {
      buffers[0].resize(fb_pixel_count);
      buffers[1].resize(fb_pixel_count);
    }
  }

  void safe_swap()
  {
    CUDA_CHECK(cudaStreamSynchronize(current_buffer().stream));
    current_buffer_index = (current_buffer_index + 1) % 2;
  }

  bool empty() const { return fb_pixel_count == 0; }

  vec2i size() const { return fb_size; }

  cudaStream_t current_stream() { return current_buffer().stream; }

  void download_async() { current_buffer().download_async(); }

  T* device_pointer() const { return current_buffer().d_pointer(); }

  T* host_pointer() const { return current_buffer().h_pointer(); }

  void deepcopy(void* dst) { current_buffer().deepcopy(dst); }

  void reset()
  {
    buffers[0].reset();
    buffers[1].reset();
  }
};

using FrameBuffer = FrameBufferTemplate<vec4f>;

}
