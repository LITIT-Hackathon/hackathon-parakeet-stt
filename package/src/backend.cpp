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

    // Transcribe a 16 kHz mono WAV file already validated on the Python side.
    std::string transcribe_file(const std::string& wav_path) {
#ifdef PARAKEET_NATIVE
        require_open();
        // The GIL is released for the whole decode: inference is long and
        // holds no Python objects, so keeping it would stall every other
        // thread in the process for the duration.
        char* out = nullptr;
        {
            py::gil_scoped_release release;
            out = parakeet_capi_transcribe_path(ctx_, wav_path.c_str(), 0);
        }
        return take(out);
#else
        (void)wav_path;
        return stub_text();
#endif
    }

    // Transcribe mono float32 PCM in [-1, 1]. The array is taken by buffer,
    // c_style + forcecast, so a contiguous float32 numpy array passes straight
    // through with no copy; anything else is coerced once by pybind rather than
    // round-tripped through a Python list.
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
            py::gil_scoped_release release;
            out = parakeet_capi_transcribe_pcm(ctx_, data, n, sample_rate, 0);
        }
        return take(out);
#else
        (void)pcm;
        (void)sample_rate;
        return stub_text();
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
    static std::string stub_text() {
        return "[stub backend] the package is wired end to end; "
               "reinstall with the default bundled build for real inference";
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
        .def("transcribe_file", &Backend::transcribe_file, py::arg("wav_path"))
        .def("transcribe_pcm", &Backend::transcribe_pcm,
             py::arg("pcm"), py::arg("sample_rate"))
        .def("close", &Backend::close);
}
