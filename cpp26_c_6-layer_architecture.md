# C++26 and C for Bioimaging: A Six-Layer Category-Theoretic Architecture

## Abstract

This document presents an architectural framework for developing high-performance bioimaging libraries using C++26 and C, organized into six layers grounded in category theory. The architecture leverages C++26's new capabilities — compile-time reflection (P2996), contracts, and `std::execution` (sender/receiver) — to encode mathematical structures from imaging research directly into type-safe, composable code. C provides the stable ABI surface for universal interoperability with Python, Julia, and the broader scientific computing ecosystem.

The six layers are: **Domain**, **Use Case**, **Category Theory**, **Port**, **Adapter**, and **Infrastructure**. Together they establish a systematic translation protocol from mathematical publications to production bioimaging libraries callable from napari, scikit-image, and PyTorch.

---

## 1. Architectural Overview

### 1.1 Layer Diagram

```
Python (napari / scikit-image / numpy / PyTorch)
          │
          ▼
┌─────────────────────────────────────────────┐
│  DRIVING PORT (Layer 4)                     │
│  extern "C" ABI + nanobind/reflection       │
│  auto-generated from port definition        │
├─────────────────────────────────────────────┤
│  ADAPTER (Layer 5)                          │
│  ndarray ↔ Image<T,N>, model format I/O     │
│  reflection-generated conversions           │
├─────────────────────────────────────────────┤
│  USE CASE (Layer 2)                         │
│  segment, denoise, register, inference      │
│  sender/receiver pipelines                  │
├─────────────────────────────────────────────┤
│  CATEGORY THEORY (Layer 3)                  │
│  Functor, Monad, Adjunction, Sheaf          │
│  concepts + contracts encoding math laws    │
├─────────────────────────────────────────────┤
│  DOMAIN (Layer 1)                           │
│  Image<T,N>, Kernel, ROI, Mask, Tensor      │
│  value objects with invariants              │
├─────────────────────────────────────────────┤
│  DRIVEN PORT (Layer 4)                      │
│  GPU compute, filesystem, network           │
│  concept-defined outbound interfaces        │
├─────────────────────────────────────────────┤
│  INFRASTRUCTURE (Layer 6)                   │
│  CUDA, BLAS/LAPACK (C), FFTW (C),          │
│  io_uring, Zarr/TIFF, std::execution       │
│  schedulers, std::simd, hazard_pointer      │
└─────────────────────────────────────────────┘
```

### 1.2 Language Placement

C and C++26 each govern specific boundaries of the architecture:

```
extern "C" ABI surface     ← C    (stable, universal FFI)
    │
Driving Port (nanobind)    ← C++26 (reflection generates both
    │                              C ABI and Python bindings)
    │
Use Case / CT / Domain     ← C++26 (concepts, contracts,
    │                              sender/receiver)
    │
Driven Port                ← C++26 (concept-constrained interfaces)
    │
Infrastructure             ← C++26 wrapping C libraries
                              (BLAS, FFTW, CUDA runtime)
```

The principle is: C at both edges (for universal interoperability and for calling foundational numerical libraries), C++26 for all the structural and mathematical thinking in between.

### 1.3 Why Two Languages

C provides a stable, universal ABI that every language on earth can call. BLAS, LAPACK, FFTW, libtiff — the foundation of scientific computing speaks C. The outermost face of a bioimaging library that the entire world calls into should be `extern "C"`.

C++26 provides the expressive power to encode mathematical structure in the type system. Preconditions become contracts. Algebraic laws become concept requirements. Composition becomes sender pipelines. The compiler catches violations of mathematical invariants at compile time. C cannot express any of this — structure exists only in comments and programmer discipline.

---

## 2. Layer 1 — Domain

The domain layer defines the core value objects of bioimaging: images, kernels, regions of interest, masks, and tensors. These types carry invariants enforced by C++26 contracts.

### 2.1 Image as a Functor

An image is a functor: `fmap` applies a function pointwise, preserving spatial structure.

```cpp
template<typename T, std::size_t N = 2>
struct Image {
    std::mdspan<T, std::dextents<std::size_t, N>> data;
    PixelSpacing<N> spacing;
    AffineTransform<N> world_transform;

    auto at(Index<N> idx) const
        pre(in_bounds(idx, data.extents()))
        -> T const&;

    auto extent(std::size_t dim) const
        pre(dim < N)
        -> std::size_t;
};

// fmap : (A → B) → Image<A,N> → Image<B,N>
template<typename F, typename T, std::size_t N>
auto fmap(F&& f, Image<T,N> const& img)
    -> Image<std::invoke_result_t<F,T>, N>;
```

### 2.2 Supporting Domain Types

```cpp
template<std::size_t N>
struct ROI {
    Index<N> origin;
    Extent<N> size;
    auto contains(Index<N> idx) const -> bool;
};

template<typename T, std::size_t N>
struct Kernel {
    Image<T,N> weights;
    Index<N> anchor;
    // Contract: anchor is within kernel bounds
};

struct Mask : Image<bool, 2> {};

template<std::size_t N>
struct PixelSpacing {
    std::array<double, N> values;
    auto isotropic() const -> bool;
};
```

### 2.3 Reflection-Driven Boilerplate Elimination

C++26 reflection eliminates the serialization, comparison, and hashing boilerplate that plagues imaging libraries. Adding a field to `Image` automatically updates JSON serialization, equality, and hash — no macro machinery required.

```cpp
// Generic serializer using reflection
template<typename T>
auto to_json(T const& obj) -> json {
    json j;
    template for (constexpr auto member : std::meta::members_of(^T)) {
        if constexpr (std::meta::is_nonstatic_data_member(member)) {
            j[std::meta::name_of(member)] =
                to_json(obj.[:member:]);
        }
    }
    return j;
}
```

---

## 3. Layer 2 — Use Cases

Use cases are application services composed as sender/receiver pipelines. Each use case is a pipeline of Kleisli arrows in the sender monad.

### 3.1 Sender/Receiver as Monadic Composition

`std::execution` provides `just` (pure/return), `then` (fmap), and `let_value` (bind/>>=). Use cases compose these into typed, async-safe pipelines:

```cpp
auto cell_segmentation(SegParams params) {
    return gaussian_blur(params.sigma)
         | threshold(params.method)
         | label_connected()
         | filter_by_area(params.min_area);
}

auto denoise_and_segment(DenoiseParams dp, SegParams sp) {
    return non_local_means(dp)
         | cell_segmentation(sp);
}
```

### 3.2 Bioimaging Use Case Catalogue

Typical use cases and their pipeline structure:

```cpp
// 3D volume segmentation
auto volume_segmentation(VolumeSegParams params) {
    return rescale_intensity()
         | anisotropic_diffusion(params.diffusion)
         | threshold_otsu()
         | morphological_closing(params.se)
         | label_connected_3d()
         | filter_by_volume(params.min_vol);
}

// Multi-channel colocalization
auto colocalization(ColocParams params) {
    return split_channels()
         | for_each_channel(background_subtract())
         | pairwise(pearson_correlation())
         | manders_coefficients();
}

// Registration pipeline
auto rigid_registration(RegParams params) {
    return compute_features()
         | match_features(params.matcher)
         | estimate_transform(params.model)
         | resample(params.interpolation);
}

// AI-augmented segmentation
auto ai_cell_analysis() {
    return gaussian_blur(1.5)
         | normalize()
         | to_tensor()
         | unet_inference(model)
         | argmax()
         | watershed()
         | measure_features();
}
```

### 3.3 Scheduler-Aware Execution

Use cases are abstract over where they run. The infrastructure layer binds schedulers:

```cpp
auto run_on_gpu(auto pipeline, auto input) {
    return just(input)
         | transfer(gpu_scheduler)
         | pipeline
         | transfer(cpu_scheduler);
}
```

---

## 4. Layer 3 — Category Theory

The category theory layer encodes the abstract mathematical structures that recur across bioimaging algorithms. This layer is the systematic translation target for mathematical publications.

### 4.1 Core Concepts

```cpp
// Functor: structure-preserving map between categories
template<template<class> class F>
concept Functor = requires(auto f, auto g, F<int> fa) {
    { fmap(f, fa) } -> std::same_as<F<decltype(f(0))>>;
    // Law: fmap(id) == id
    // Law: fmap(f . g) == fmap(f) . fmap(g)
};

// Monad: Functor with sequencing
template<template<class> class M>
concept Monad = Functor<M> && requires(auto a, auto f) {
    { pure<M>(a) } -> std::same_as<M<decltype(a)>>;
    { bind(std::declval<M<int>>(), f) };
    // Law: bind(pure(a), f) == f(a)           (left identity)
    // Law: bind(m, pure) == m                  (right identity)
    // Law: bind(bind(m,f), g) == bind(m, λx.bind(f(x),g))
};

// Adjunction: pair of functors with unit/counit
template<typename F, typename G>
concept Adjunction = requires(auto a, auto b) {
    { unit(a) };     // η: Id → G∘F
    { counit(b) };   // ε: F∘G → Id
    // hom(F(a), b) ≅ hom(a, G(b))
};

// MonoidAction: parameterized family acting on objects
template<typename M, typename X>
concept MonoidAction = Monoid<M> && requires(M m, X x) {
    { act(m, x) } -> std::same_as<X>;
    // act(e, x) == x
    // act(m*n, x) == act(m, act(n, x))
};

// Presheaf: contravariant functor (local data on open sets)
template<typename F>
concept Presheaf = requires(F f, OpenSet u, OpenSet v) {
    { f(u) } -> VectorSpace;
    { f.restrict(u, v) } -> LinearMap;
};

// Sheaf: presheaf satisfying gluing axiom
template<typename F>
concept Sheaf = Presheaf<F>;
```

### 4.2 Kleisli Composition for Pipelines

Processing steps are Kleisli arrows in the sender monad. Composition is the fundamental operation:

```cpp
// An imaging step: Image<A> → sender_of<Image<B>>
template<typename F>
concept ImagingStep =
    requires(F f, Image<float,3> img) {
        { f(img) } -> SenderOf<Image<auto, 3>>;
    };

// Kleisli composition: (>=>) for imaging steps
auto operator|(ImagingStep auto a, ImagingStep auto b) {
    return [=](auto img) {
        return just(img)
            | let_value(a)
            | let_value(b);
    };
}
```

### 4.3 Paper-to-Code Translation Table

When reading a mathematical publication, identify which categorical structure appears, then use the corresponding C++26 encoding:

| Publication uses...                                     | CT structure               | C++26 encoding                             |
|---------------------------------------------------------|----------------------------|--------------------------------------------|
| Transform pairs (erode/dilate, blur/sharpen)            | Adjunction                 | Concept pair + unit/counit contracts       |
| Parameterized family Tₜ (diffusion, flow, scale)       | Monoid action              | Monoid concept + `act()`                   |
| Filtered construction (persistence, wavelets)           | Functor                    | Concept + fmap                             |
| Local-to-global (fusion, registration, mosaicing)       | Sheaf                      | Presheaf concept + gluing contract         |
| Reversible transforms (FFT, wavelet)                    | Natural isomorphism        | Concept with invertibility contract        |
| Probabilistic / uncertain output                        | Monad                      | sender/receiver pipeline                   |
| Multi-step algorithm pipeline                           | Kleisli composition        | `let_value` chains                         |
| Neural network layer                                    | Endofunctor                | Layer concept + forward()                  |
| Loss function                                           | Natural transformation     | Concept mapping prediction → scalar        |
| Optimizer step                                          | Endomorphism               | Params → Params monoid                     |

---

## 5. Layer 4 — Ports

Ports define the boundaries of the system. Driving ports face inward from external callers (Python, CLI, GUI). Driven ports face outward toward infrastructure (GPU, filesystem, network).

### 5.1 Driving Port (Inbound)

The driving port is the API surface that Python calls into. It is flat, explicit, and devoid of categorical abstraction — the consumer sees only concrete operations.

```cpp
struct ImagingPort {
    auto segment_cells(
        Image<float,3> const& volume,
        SegParams params
    ) -> sender_of<Image<uint32_t,3>> auto;

    auto denoise(
        Image<float,2> const& img,
        DenoiseMethod method
    ) -> sender_of<Image<float,2>> auto;

    auto register_images(
        Image<float,2> const& fixed,
        Image<float,2> const& moving,
        RegParams params
    ) -> sender_of<AffineTransform<2>> auto;

    auto run_inference(
        Image<float,3> const& volume,
        ModelPath model
    ) -> sender_of<Image<uint32_t,3>> auto;

    auto compute_persistence(
        Image<float,3> const& volume,
        float threshold_range
    ) -> sender_of<PersistenceDiagram> auto;
};
```

### 5.2 Driven Port (Outbound)

Driven ports are concept-constrained interfaces to external resources:

```cpp
template<typename R, typename T, std::size_t N>
concept ImageRepository =
    requires(R r, Image<T,N> img, std::string id) {
        { r.save(id, img) } -> SenderOf<void>;
        { r.load(id) }      -> SenderOf<Image<T,N>>;
    };

template<typename G>
concept GpuCompute =
    requires(G g, Tensor t, KernelSpec spec) {
        { g.dispatch(spec, t) } -> SenderOf<Tensor>;
        { g.scheduler() }       -> Scheduler;
    };

template<typename M>
concept ModelLoader =
    requires(M m, ModelPath path) {
        { m.load(path) }        -> SenderOf<Model>;
        { m.infer(Model{}, Tensor{}) } -> SenderOf<Tensor>;
    };
```

### 5.3 C ABI Surface

For maximum interoperability beyond Python (Julia, MATLAB, LabVIEW, ImageJ plugins), the driving port exposes a thin `extern "C"` layer:

```c
/* bioimg.h — stable C ABI */
#ifdef __cplusplus
extern "C" {
#endif

typedef struct bioimg_image bioimg_image;
typedef struct bioimg_params bioimg_params;

int bioimg_segment_cells(
    const bioimg_image* volume,
    const bioimg_params* params,
    bioimg_image** result
);

int bioimg_denoise(
    const bioimg_image* img,
    int method,
    bioimg_image** result
);

void bioimg_image_free(bioimg_image* img);

/* Returns pointer to raw data for zero-copy access */
const float* bioimg_image_data(const bioimg_image* img);
int bioimg_image_ndim(const bioimg_image* img);
const int64_t* bioimg_image_shape(const bioimg_image* img);

#ifdef __cplusplus
}
#endif
```

---

## 6. Layer 5 — Adapters

Adapters convert between external representations and domain types. C++26 reflection automates most of this.

### 6.1 Reflection-Generated Python Bindings

```cpp
template<typename Port>
void bind_port(nb::module_& m) {
    template for (constexpr auto fn : std::meta::members_of(^Port)) {
        if constexpr (std::meta::is_public(fn)
                   && std::meta::is_function(fn)) {
            m.def(
                std::meta::name_of(fn),
                [:fn:]
            );
        }
    }
}

NB_MODULE(bioimg, m) {
    bind_ndarray_conversions<float, uint8_t, uint32_t>(m);
    bind_port<ImagingPort>(m);
}
```

### 6.2 Zero-Copy ndarray Conversion

nanobind's `nb::ndarray` wraps numpy buffers directly into `Image<T,N>` via `std::mdspan` — no memory copy:

```cpp
template<typename T, std::size_t N>
auto from_ndarray(nb::ndarray<T, nb::ndim<N>> arr) -> Image<T,N> {
    // Zero-copy: mdspan wraps the numpy buffer pointer
    return Image<T,N>{
        .data = std::mdspan<T, std::dextents<std::size_t, N>>(
            arr.data(),
            to_extents<N>(arr)
        ),
        .spacing = PixelSpacing<N>::isotropic(1.0)
    };
}
```

### 6.3 File Format Adapters

```cpp
// Reflection-generated TIFF adapter
template<Aggregate T>
class TiffAdapter : public ImageRepository<T, 2> {
    auto save(std::string id, Image<T,2> const& img)
        -> sender_of<void> auto
    {
        constexpr auto meta = std::meta::members_of(^Image<T,2>);
        return on(io_scheduler, [=] {
            // libtiff C API call — infrastructure layer
            write_tiff(id, img.data, img.spacing);
        });
    }
};

// Zarr adapter for large volumetric data
template<typename T, std::size_t N>
class ZarrAdapter : public ImageRepository<T, N> {
    auto load(std::string id) -> sender_of<Image<T,N>> auto {
        return on(io_scheduler, [=] {
            return read_zarr_chunk<T,N>(id);
        });
    }
};
```

---

## 7. Layer 6 — Infrastructure

Infrastructure provides concrete implementations of driven ports, using C libraries for foundational computation and C++26 for scheduling and composition.

### 7.1 C Library Integration

The infrastructure layer wraps battle-tested C libraries:

```cpp
// FFTW (C) wrapper
namespace infra {
    auto fft_2d(Image<float,2> const& img)
        -> Image<std::complex<float>, 2>
    {
        auto out = allocate_complex(img.extent(0), img.extent(1));
        // Call C function directly
        fftwf_plan plan = fftwf_plan_dft_r2c_2d(
            img.extent(0), img.extent(1),
            const_cast<float*>(img.data.data_handle()),
            reinterpret_cast<fftwf_complex*>(out.data.data_handle()),
            FFTW_ESTIMATE
        );
        fftwf_execute(plan);
        fftwf_destroy_plan(plan);
        return out;
    }

    // BLAS (C) for linear algebra in registration
    auto gemm(MatrixView<double> A, MatrixView<double> B)
        -> Matrix<double>
    {
        Matrix<double> C(A.rows(), B.cols());
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    A.rows(), B.cols(), A.cols(),
                    1.0, A.data(), A.stride(),
                    B.data(), B.stride(),
                    0.0, C.data(), C.stride());
        return C;
    }
}
```

### 7.2 Scheduler Configuration

`std::execution` schedulers bind abstract pipelines to hardware:

```cpp
struct InfraContext {
    // CPU thread pool for compute-bound work
    static_thread_pool cpu_pool{std::thread::hardware_concurrency()};

    // I/O scheduler for file and network operations
    io_uring_scheduler io_sched;

    // GPU scheduler for CUDA work
    cuda_scheduler gpu_sched;

    auto cpu()  -> auto { return cpu_pool.get_scheduler(); }
    auto io()   -> auto { return io_sched; }
    auto gpu()  -> auto { return gpu_sched; }
};
```

### 7.3 Concurrency Primitives

C++26 provides lock-free data structures for concurrent imaging pipelines:

```cpp
// Hazard pointers for concurrent tile cache
// (multiple threads processing image tiles simultaneously)
template<typename T>
class ConcurrentTileCache {
    std::hazard_pointer<TileNode<T>> head_;
public:
    auto get(TileIndex idx) -> std::optional<Image<T,2>>;
    auto insert(TileIndex idx, Image<T,2> tile) -> void;
};
```

---

## 8. Category Theory in Practice: Bioimaging Algorithms

### 8.1 Mathematical Morphology — Adjunctions

Erosion and dilation form an adjoint pair (Galois connection) on the lattice of images. All derived morphological operations are compositions of this single adjunction:

```cpp
struct Dilate {
    StructElem se;
    auto operator()(Image<bool,2> const& img) -> Image<bool,2>;
};

struct Erode {
    StructElem se;
    auto operator()(Image<bool,2> const& img) -> Image<bool,2>;
};

static_assert(Adjunction<Dilate, Erode>);

// Derived operations from the adjunction:
auto opening(StructElem se) {
    return erode(se) | dilate(se);    // counit: ε = F∘G → Id
}
auto closing(StructElem se) {
    return dilate(se) | erode(se);    // unit: η = Id → G∘F
}
auto white_top_hat(StructElem se) {
    return [=](auto img) { return img - opening(se)(img); };
}
auto morphological_gradient(StructElem se) {
    return [=](auto img) { return dilate(se)(img) - erode(se)(img); };
}
```

### 8.2 Persistent Homology — Functors to Vec

The persistence module is a functor from a filtration (poset) category to vector spaces:

```cpp
template<typename F>
concept PersistenceFunctor =
    Functor<F> &&
    requires(F f, float threshold_a, float threshold_b) {
        { f.at(threshold_a) }                  -> VectorSpace;
        { f.morphism(threshold_a, threshold_b)} -> LinearMap;
    };

auto sublevel_persistence(Image<float,3> const& vol)
    -> PersistenceDiagram
{
    auto filtration = sublevel_filtration(vol);
    auto F = compute_persistence_functor(filtration);
    return diagram_from(F);
}
```

### 8.3 Scale Space — Monoid Actions

Gaussian scale space is a monoid homomorphism from (ℝ≥0, +) to End(Image):

```cpp
struct ScaleParam {
    float sigma;
    friend ScaleParam operator*(ScaleParam a, ScaleParam b) {
        return {std::sqrt(a.sigma*a.sigma + b.sigma*b.sigma)};
    }
    static ScaleParam identity() { return {0.f}; }
};

static_assert(Monoid<ScaleParam>);

auto gaussian_scale_space(Image<float,2> const& img, ScaleParam s)
    -> Image<float,2>;

// Any diffusion paper (Perona-Malik, Weickert coherence-enhancing)
// implements this SAME concept with a different act()
```

### 8.4 Multi-Modal Fusion — Sheaves

Sheaf theory encodes local-to-global consistency for multi-modal imaging:

```cpp
auto fuse_modalities(
    Sheaf auto const& measurements,
    CoverGraph const& overlap
) -> Image<float, 3>
{
    // Minimize inconsistency across overlapping regions
    // using the sheaf Laplacian
    auto L = sheaf_laplacian(measurements, overlap);
    return solve_least_squares(L);
}
```

### 8.5 AI Integration — Neural Networks as Functors

Neural network layers are endofunctors on tensor spaces. The categorical framework unifies classical imaging and AI:

```cpp
template<typename L>
concept Layer = requires(L l, Tensor x) {
    { l.forward(x) } -> TensorLike;
    { l.params() }   -> RangeOf<Parameter>;
};

// Classical and AI steps compose identically:
auto cell_analysis = gaussian_blur(1.5)    // classical (Kleisli arrow)
                   | normalize()            // classical
                   | to_tensor()            // adapter
                   | unet_inference(model)  // AI (same arrow type)
                   | argmax()               // post-processing
                   | watershed()            // classical
                   | measure_features();    // classical
```

---

## 9. Python Integration

### 9.1 Driving Port via nanobind

nanobind provides faster compilation (~4×), smaller binaries (~5×), and lower runtime overhead (~10×) compared to pybind11. Its `nb::ndarray` supports NumPy, PyTorch, JAX, and TensorFlow through a single C++ type.

C++26 reflection auto-generates bindings from the driving port definition. Adding a method to `ImagingPort` and recompiling makes it appear in Python automatically — no binding file to maintain.

### 9.2 Python Usage

```python
import bioimg
import numpy as np
from skimage.io import imread

# numpy arrays go in, numpy arrays come out (zero-copy)
vol = imread("cells.tif").astype(np.float32)
labels = bioimg.segment_cells(vol, sigma=1.5, min_area=100)

# Directly usable in napari
import napari
viewer = napari.Viewer()
viewer.add_image(vol)
viewer.add_labels(labels)

# AI + classical — same API
ai_labels = bioimg.run_inference(vol, model="cell_unet.onnx")
```

### 9.3 C ABI for Other Languages

The `extern "C"` surface enables integration beyond Python:

```julia
# Julia via ccall
result = ccall((:bioimg_segment_cells, "libbioimg"),
               Cint, (Ptr{BioImgImage}, Ptr{BioImgParams}, Ptr{Ptr{BioImgImage}}),
               volume, params, result_ptr)
```

```matlab
% MATLAB via loadlibrary
loadlibrary('libbioimg', 'bioimg.h');
calllib('libbioimg', 'bioimg_denoise', img_ptr, method, result_ptr);
```

---

## 10. C++26 Feature Mapping

A summary of how each C++26 feature serves the architecture:

| C++26 Feature              | Layer(s)               | Role in Bioimaging                                                 |
|----------------------------|------------------------|--------------------------------------------------------------------|
| Contracts (pre/post)       | Domain, Port           | Enforce tensor shapes, pixel ranges, domain invariants             |
| Reflection (P2996)         | Adapter, Port          | Auto-generate bindings, serialization, format converters           |
| std::execution             | Use Case, Infra        | Compose async pipelines, schedule across CPU/GPU/IO                |
| Concepts                   | CT, Port, Domain       | Encode Functor, Monad, Adjunction; constrain port interfaces       |
| std::mdspan                | Domain                 | Zero-copy multi-dimensional image views                            |
| std::simd                  | Infrastructure         | Vectorized pixel operations                                        |
| constexpr / consteval      | CT, Domain             | Compile-time shape arithmetic, type-level dimension checking       |
| Hazard pointers / RCU      | Infrastructure         | Lock-free concurrent tile caches, parallel image processing        |
| Bounds-hardened stdlib      | All                    | Safety across all layers                                           |

---

## 11. C vs C++26 Decision Matrix

| Criterion                          | C                  | C++26                     |
|------------------------------------|--------------------|---------------------------|
| ABI stability                      | Universal, stable  | Name-mangled, unstable    |
| Cross-language FFI                 | Every language      | Requires wrappers         |
| Category theory encoding           | Impossible         | Concepts + contracts      |
| Generic programming                | void* + macros     | Templates + concepts      |
| Composition                        | Function pointers  | sender/receiver, Kleisli  |
| Compile-time computation           | #define            | constexpr + reflection    |
| Invariant enforcement              | assert()           | pre/post contracts        |
| Memory safety                      | Manual             | RAII, bounds hardening    |
| Existing math library ecosystem    | BLAS, FFTW, libtiff| Wraps C libraries         |
| Runtime performance                | Excellent          | Excellent (zero-overhead) |

Use C for: the outermost ABI surface, calling into existing C math libraries, embedding in other languages.

Use C++26 for: all internal design, categorical abstraction, type-safe composition, async scheduling, auto-generated bindings.

---

## 12. From Paper to Library: The Workflow

The architecture establishes a systematic process for translating mathematical publications into production libraries:

1. **Read the paper.** Identify the core mathematical structures (transform families, adjoint pairs, filtrations, local-to-global constructions).

2. **Map to CT.** Consult the translation table (Section 4.3). Determine if the paper's algorithm is best modeled as an adjunction, functor, monoid action, sheaf, or Kleisli composition.

3. **Check the CT layer.** If the categorical concept already exists, instantiate it with the paper's concrete types. If not, define a new concept with appropriate contracts encoding the paper's axioms.

4. **Implement in the Domain layer.** Define the value objects the paper requires (specialized image types, kernels, parameters).

5. **Compose in the Use Case layer.** Build the paper's algorithm as a pipeline of Kleisli arrows over existing and new imaging steps.

6. **Port auto-generates.** Add the new use case to `ImagingPort`. Recompile. Reflection generates both the C ABI header and the nanobind Python binding automatically.

7. **Scientist calls from Python.** The new algorithm appears in napari, Jupyter, or scikit-image workflows with zero additional integration work.

---

## 13. Conclusion

C++26 and C together provide a complete stack for bioimaging library development. C anchors the system at its edges — stable ABI for universal reach, foundational numerical libraries for proven performance. C++26 fills the interior with mathematical structure: concepts encode categorical laws, contracts enforce domain invariants, reflection eliminates boilerplate, and `std::execution` composes heterogeneous computation across CPU, GPU, and I/O.

The six-layer architecture ensures that mathematical publications translate systematically into typed, composable, production-grade code. The category theory layer is not ornamental — it is the mechanism by which new algorithms slot into existing infrastructure without ad hoc integration. The driving port ensures that the entire library is accessible from Python with zero maintenance burden.

The result is bioimaging libraries that are as principled as Haskell, as fast as C, and as accessible as Python.
