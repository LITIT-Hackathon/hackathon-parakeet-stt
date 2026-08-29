// Native backend for parakeet-stt.
//
// Two builds come out of this one file:
//
//   native  (-DPARAKEET_NATIVE, default)  calls parakeet.cpp's flat C-API
//   stub    (PARAKEET_STT_BUNDLED=OFF)    compiles and returns canned text
//
// The stub lets the package build, install, test and demo with no engine and
// no model. Both expose the identical Python surface, so switching is a CMake
// flag, never a code change upstream of here.
//
// transcribe_pcm returns a JSON document (a one-element array), not a bare
// string: the *_json C-API entry point yields per-word timestamps and
// confidence at no extra decode cost, so the Python layer gets both the
// transcript and the word spans from one call.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>

#ifdef PARAKEET_NATIVE
extern "C" {
#include "parakeet_capi.h"
}
#endif

namespace py = pybind11;

namespace {

class Backend {
public:
    explicit Backend(const std::string& model_path) : model_path_(model_path) {
#ifdef PARAKEET_NATIVE
        ctx_ = parakeet_capi_load(model_path.c_str());
        if (!ctx_) {
            throw std::runtime_error("failed to load model: " + model_path);
        }
#endif
    }

    ~Backend() { close(); }

    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;

    void close() {
#ifdef PARAKEET_NATIVE
        if (ctx_) {
            parakeet_capi_free(ctx_);
            ctx_ = nullptr;
        }
#endif
    }

    // Transcribe mono float32 PCM in [-1, 1]. Returns a JSON document:
    //   [{"text": "...", "frame_sec": F,
    //     "words":  [{"w": "...", "start": S, "end": E, "conf": C}, ...],
    //     "tokens": [...]}]
    // The array has one element (this is the batch entry point called with
    // n_clips = 1). The Python layer parses text + words out of it.
    std::string transcribe_pcm(
        py::array_t<float, py::array::c_style | py::array::forcecast> pcm,
        int sample_rate) {
#ifdef PARAKEET_NATIVE
        require_open();
        py::buffer_info buf = pcm.request();   // touches Python, before the release
        const float* data = static_cast<const float*>(buf.ptr);
        const int n = static_cast<int>(buf.size);
        char* out = nullptr;
        {
            // Long, holds no Python objects: release the GIL for the decode.
            py::gil_scoped_release release;
            out = parakeet_capi_transcribe_pcm_batch_json(
                ctx_, data, &n, /*n_clips=*/1, sample_rate, /*decoder=*/0);
        }
        return take(out);
#else
        (void)pcm;
        (void)sample_rate;
        return stub_json();
#endif
    }

    static std::string backend_name() {
#ifdef PARAKEET_NATIVE
        return "parakeet.cpp";
#else
        return "stub";
#endif
    }

    static bool is_native() {
#ifdef PARAKEET_NATIVE
        return true;
#else
        return false;
#endif
    }

private:
    std::string model_path_;

#ifdef PARAKEET_NATIVE
    parakeet_ctx* ctx_ = nullptr;

    void require_open() const {
        if (!ctx_) throw std::runtime_error("backend is closed");
    }

    // The C-API hands back a malloc'd string that we own and must free.
    std::string take(char* p) {
        if (!p) {
            const char* err = parakeet_capi_last_error(ctx_);
            throw std::runtime_error(err ? err : "transcription failed");
        }
        std::string s(p);
        parakeet_capi_free_string(p);
        return s;
    }
#else
    static std::string stub_json() {
        return R"([{"text":"[stub backend] the package is wired end to end; )"
               R"(reinstall with the default bundled build for real inference",)"
               R"("frame_sec":0.08,"words":[],"tokens":[]}])";
    }
#endif
};

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "Native inference core for parakeet-stt.";

    m.def("backend_name", &Backend::backend_name,
          "Which backend this extension was compiled against.");
    m.def("is_native", &Backend::is_native,
          "True when built against parakeet.cpp, False for the stub.");

    py::class_<Backend>(m, "Backend")
        .def(py::init<const std::string&>(), py::arg("model_path"))
        .def("transcribe_pcm", &Backend::transcribe_pcm,
             py::arg("pcm"), py::arg("sample_rate"))
        .def("close", &Backend::close);
}
