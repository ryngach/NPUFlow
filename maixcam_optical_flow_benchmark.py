"""
Optical Flow Benchmark for MaixCAM
Direct libcviruntime.so + MaixPy Camera
Author: Ruslan Ryngach
2025
"""
import ctypes
import numpy as np
from maix import camera, image, time, display
import os
import sys

# Спробуємо імпортувати image2cv (доступно на старіших версіях MaixPy)
try:
    from maix.image import image2cv
    HAS_IMAGE2CV = True
except ImportError:
    HAS_IMAGE2CV = False

# Спробуємо використати cv2 для швидкої візуалізації
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: cv2 not available, using slower numpy visualization")

# Python C API для роботи з PyCapsule
pythonapi = ctypes.pythonapi
pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]


class CVI_SHAPE(ctypes.Structure):
    _fields_ = [
        ("dim_size", ctypes.c_uint32),
        ("dim", ctypes.c_int32 * 8),
    ]

class CVI_TENSOR(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("fmt", ctypes.c_int32),
        ("_pad1", ctypes.c_int32),
        ("shape", CVI_SHAPE),
        ("stride", CVI_SHAPE),
        ("pixel_format", ctypes.c_uint64),
        ("count", ctypes.c_uint64),
        ("mem_type", ctypes.c_int32),
        ("_pad2", ctypes.c_int32),
        ("paddr", ctypes.c_uint64),
        ("vaddr", ctypes.c_void_p),
        ("aligned", ctypes.c_int32),
        ("qscale", ctypes.c_float),
        ("mean", ctypes.c_float * 3),
        ("_pad3", ctypes.c_int32),
        ("scale", ctypes.c_float * 3),
        ("zero_point", ctypes.c_int32),
    ]

CVI_FMT_INT8 = 1
CVI_FMT_UINT8 = 2


class OpticalFlowBenchmark:
    def __init__(self, model_path, warmup=10, benchmark=100, visualize=False):
        """Initialize optical flow benchmark.
        
        Args:
            model_path: Path to .cvimodel file
            warmup: Number of warmup iterations
            benchmark: Number of benchmark iterations
            visualize: Enable visualization on display
        """
        self.model_path = model_path
        self.warmup_times = warmup
        self.benchmark_times = benchmark
        self.visualize = visualize
        
        print(f"\n{'='*60}")
        print("Optical Flow Benchmark v4.0")
        print(f"{'='*60}")
        print(f"Model: {os.path.basename(model_path)}")
        
        print("\nLoading libcviruntime.so...")
        self.lib = ctypes.CDLL('/usr/lib/libcviruntime.so')
        self._setup_functions()
        print("✓ Library loaded")
        
        print("\nLoading model...")
        self.model_handle = ctypes.c_void_p()
        ret = self.lib.CVI_NN_RegisterModel(
            model_path.encode('utf-8'),
            ctypes.byref(self.model_handle)
        )
        if ret != 0:
            raise RuntimeError(f"RegisterModel failed: {ret}")
        print("✓ Model loaded")
        
        print("\nGetting tensors...")
        self.inputs = ctypes.POINTER(CVI_TENSOR)()
        self.outputs = ctypes.POINTER(CVI_TENSOR)()
        self.input_num = ctypes.c_int32()
        self.output_num = ctypes.c_int32()
        
        ret = self.lib.CVI_NN_GetInputOutputTensors(
            self.model_handle,
            ctypes.byref(self.inputs),
            ctypes.byref(self.input_num),
            ctypes.byref(self.outputs),
            ctypes.byref(self.output_num)
        )
        if ret != 0:
            raise RuntimeError(f"GetTensors failed: {ret}")
        print(f"✓ Found {self.input_num.value} inputs, {self.output_num.value} outputs")
        
        self._analyze_model()
        
        if HAS_CV2:
            print("Allocating visualization buffers...")
            # Pre-allocate buffers for visualization to avoid repeated allocations
            self.vis_buf_u8_1 = np.zeros((self.height, self.width), dtype=np.uint8)
            self.vis_buf_u8_2 = np.zeros((self.height, self.width), dtype=np.uint8)
            self.vis_buf_u8_3 = np.zeros((self.height, self.width), dtype=np.uint8)
            # RGB buffer for final image
            self.vis_rgb_buf = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        print("\nPreparing buffers...")
        self.frame1_np = np.zeros((3, self.height, self.width), dtype=self.buffer_dtype)
        self.frame2_np = np.zeros((3, self.height, self.width), dtype=self.buffer_dtype)
        print(f"✓ Buffers: {self.frame1_np.shape}, dtype={self.buffer_dtype}")
        
        # Кешуємо ctypes references для інференсу
        self.input0_ref = ctypes.byref(self.inputs[0])
        self.input1_ref = ctypes.byref(self.inputs[1])
        
        print("\n✓ Initialization complete!")
        
        # Дисплей ініціалізуємо лише коли він справді потрібен
        self.disp = None
    
    def _setup_functions(self):
        """Setup ctypes function signatures."""
        self.lib.CVI_NN_RegisterModel.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.CVI_NN_RegisterModel.restype = ctypes.c_int
        
        self.lib.CVI_NN_GetInputOutputTensors.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.POINTER(CVI_TENSOR)),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.POINTER(CVI_TENSOR)),
            ctypes.POINTER(ctypes.c_int32),
        ]
        self.lib.CVI_NN_GetInputOutputTensors.restype = ctypes.c_int
        
        self.lib.CVI_NN_SetTensorPtr.argtypes = [ctypes.POINTER(CVI_TENSOR), ctypes.c_void_p]
        self.lib.CVI_NN_SetTensorPtr.restype = None
        
        self.lib.CVI_NN_Forward.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(CVI_TENSOR),
            ctypes.c_int32,
            ctypes.POINTER(CVI_TENSOR),
            ctypes.c_int32,
        ]
        self.lib.CVI_NN_Forward.restype = ctypes.c_int
        
        self.lib.CVI_NN_CleanupModel.argtypes = [ctypes.c_void_p]
        self.lib.CVI_NN_CleanupModel.restype = ctypes.c_int
        
        self.lib.CVI_NN_TensorCount.argtypes = [ctypes.POINTER(CVI_TENSOR)]
        self.lib.CVI_NN_TensorCount.restype = ctypes.c_size_t
        
        self.lib.CVI_NN_TensorQuantScale.argtypes = [ctypes.POINTER(CVI_TENSOR)]
        self.lib.CVI_NN_TensorQuantScale.restype = ctypes.c_float
        
        self.lib.CVI_NN_TensorPtr.argtypes = [ctypes.POINTER(CVI_TENSOR)]
        self.lib.CVI_NN_TensorPtr.restype = ctypes.c_void_p
    
    def _analyze_model(self):
        """Analyze model to determine preprocessing requirements."""
        input0 = self.inputs[0]
        
        count = self.lib.CVI_NN_TensorCount(ctypes.byref(input0))
        qscale = self.lib.CVI_NN_TensorQuantScale(ctypes.byref(input0))
        fmt = input0.fmt
        
        print(f"Format: {fmt} (INT8={CVI_FMT_INT8}, UINT8={CVI_FMT_UINT8})")
        print(f"Count: {count}")
        print(f"Qscale: {qscale}")
        
        if abs(qscale - 1.0) < 0.1:
            self.has_fuse = True
            self.buffer_dtype = np.uint8
            print("✓ WITH fuse_preprocess (direct copy)")
        elif abs(qscale - 127.0) < 5.0:
            self.has_fuse = False
            self.buffer_dtype = np.int8
            print("✓ WITHOUT fuse_preprocess (qscale ≈ 127)")
        else:
            self.has_fuse = False
            self.buffer_dtype = np.int8
            print(f"✓ WITHOUT fuse_preprocess (qscale = {qscale})")
        
        if count == 245760:
            self.height, self.width = 256, 320
        elif count == 921600:
            self.height, self.width = 480, 640
        else:
            raise RuntimeError(f"Unknown dimensions for count={count}")
        print(f"✓ Dimensions: {self.width}x{self.height}")
        
        # Кешуємо параметри для оптимізації
        self.buffer_size = self.height * self.width * 3
        if not self.has_fuse:
            self.qscale = qscale
            if abs(qscale - 127.0) < 1.0:
                self.use_bitshift = True
            else:
                self.use_bitshift = False
                self.scale_factor = qscale / 255.0
        
        # Перевіряємо чи підтримується zero-copy через data()
        print(f"\nInitializing camera ({self.width}x{self.height})...")
        self.cam = camera.Camera(self.width, self.height, image.Format.FMT_RGB888)
        test_img = self.cam.read()
        self.use_zerocopy = hasattr(test_img, 'data')
        
        # Визначаємо метод preprocessing
        if self.use_zerocopy:
            self.preprocess_method = "zero-copy"
            print("✓ Camera initialized (zero-copy mode)")
        elif HAS_IMAGE2CV:
            self.preprocess_method = "image2cv"
            print("✓ Camera initialized (image2cv mode)")
        else:
            self.preprocess_method = "to_bytes"
            print("✓ Camera initialized (to_bytes fallback mode)")
    
    def _preprocess(self, img, out_buffer):
        """Preprocess image for model input (auto-detect best method).
        
        Args:
            img: MaixPy image
            out_buffer: Output numpy array (C, H, W)
        """
        if self.preprocess_method == "zero-copy":
            # Zero-copy метод через PyCapsule (найшвидший, новіші версії MaixPy)
            capsule = img.data()
            data_ptr = pythonapi.PyCapsule_GetPointer(capsule, None)
            buffer_from_memory = (ctypes.c_uint8 * self.buffer_size).from_address(data_ptr)
            img_flat = np.frombuffer(buffer_from_memory, dtype=np.uint8)
        elif self.preprocess_method == "image2cv":
            # Метод через image2cv (оптимізований на старіших версіях MaixPy)
            img_array = image2cv(img, copy=False)
            img_flat = img_array.ravel()
        else:
            # Fallback метод через to_bytes()
            img_bytes = img.to_bytes()
            img_flat = np.frombuffer(img_bytes, dtype=np.uint8)
        
        if self.has_fuse:
            # Оптимізована версія для fuse_preprocess
            img_hwc = img_flat.reshape((self.height, self.width, 3))
            # Швидке копіювання по каналах через numpy slicing
            out_buffer[0] = img_hwc[:, :, 0]
            out_buffer[1] = img_hwc[:, :, 1]
            out_buffer[2] = img_hwc[:, :, 2]
        else:
            # Для моделей без fuse_preprocess
            img_hwc = img_flat.reshape((self.height, self.width, 3))
            
            if self.use_bitshift:
                # Оптимізований метод: бітовий зсув
                out_buffer[0] = (img_hwc[:, :, 0] >> 1).astype(np.int8)
                out_buffer[1] = (img_hwc[:, :, 1] >> 1).astype(np.int8)
                out_buffer[2] = (img_hwc[:, :, 2] >> 1).astype(np.int8)
            else:
                # Векторизоване масштабування
                out_buffer[0] = (img_hwc[:, :, 0] * self.scale_factor).astype(np.int8)
                out_buffer[1] = (img_hwc[:, :, 1] * self.scale_factor).astype(np.int8)
                out_buffer[2] = (img_hwc[:, :, 2] * self.scale_factor).astype(np.int8)
    
    def _inference(self):
        """Run model inference (optimized version)."""
        # Оновлюємо вказівники (вони можуть змінитись після swap)
        self.lib.CVI_NN_SetTensorPtr(
            self.input0_ref, 
            self.frame1_np.ctypes.data_as(ctypes.c_void_p)
        )
        self.lib.CVI_NN_SetTensorPtr(
            self.input1_ref, 
            self.frame2_np.ctypes.data_as(ctypes.c_void_p)
        )
        
        ret = self.lib.CVI_NN_Forward(
            self.model_handle,
            self.inputs,
            self.input_num.value,
            self.outputs,
            self.output_num.value
        )
        if ret != 0:
            raise RuntimeError(f"Forward failed: {ret}")
    
    def _get_flow(self):
        """Get optical flow output from model.
        
        Returns:
            numpy array with shape (2, H, W) - flow_x and flow_y
        """
        try:
            output0 = self.outputs[0]
            
            # Отримуємо розмір вихідних даних
            output_count = self.lib.CVI_NN_TensorCount(ctypes.byref(output0))
            
            # Перевіряємо розмір
            if output_count != 2 * self.height * self.width:
                raise RuntimeError(f"Unexpected output count: {output_count} != {2 * self.height * self.width}")
            
            # Отримуємо pointer до даних через API
            data_ptr = self.lib.CVI_NN_TensorPtr(ctypes.byref(output0))
            
            if data_ptr == 0 or data_ptr is None:
                raise RuntimeError(f"Invalid data pointer: {data_ptr}")
            
            # Output завжди float32 (незалежно від fmt field в структурі)
            # Модель називається flow_full_Add_f32 - це вказує на float32 output
            buffer = (ctypes.c_float * output_count).from_address(data_ptr)
            flow_data = np.frombuffer(buffer, dtype=np.float32)
            
            # Reshape до (2, H, W)
            flow = flow_data.reshape((2, self.height, self.width))
            return flow
        except Exception as e:
            print(f"Error in _get_flow: {e}")
            # Повертаємо нульовий flow в разі помилки
            return np.zeros((2, self.height, self.width), dtype=np.float32)
    
    def _visualize_flow(self, img, flow):
        """Visualize optical flow on image (cv2 optimized if available).
        
        Args:
            img: Original MaixPy image
            flow: Flow array (2, H, W)
        
        Returns:
            Image with flow visualization or original image on error
        """
        try:
            if HAS_CV2:
                # Use cv2 for faster processing with pre-allocated buffers
                fx = flow[0]
                fy = flow[1]
                
                # 1. Magnitude (Manhattan) - optimized
                # Use pre-allocated buffers to avoid allocation overhead
                # buf1 = |fx|, buf2 = |fy|
                cv2.convertScaleAbs(fx, dst=self.vis_buf_u8_1)
                cv2.convertScaleAbs(fy, dst=self.vis_buf_u8_2)
                
                # buf3 = |fx| + |fy| (Magnitude)
                cv2.add(self.vis_buf_u8_1, self.vis_buf_u8_2, dst=self.vis_buf_u8_3)
                mag = self.vis_buf_u8_3
                
                # 2. Color mapping
                # buf1 = saturate(fx*3 + 127)
                cv2.addWeighted(fx, 3.0, fx, 0, 127.0, dtype=cv2.CV_8U, dst=self.vis_buf_u8_1)
                # buf2 = saturate(fy*3 + 127)
                cv2.addWeighted(fy, 3.0, fy, 0, 127.0, dtype=cv2.CV_8U, dst=self.vis_buf_u8_2)
                
                # 3. Masking (Zero flow = black)
                # buf1 = buf1 * mag / 255
                cv2.multiply(self.vis_buf_u8_1, mag, scale=1.0/255.0, dtype=cv2.CV_8U, dst=self.vis_buf_u8_1)
                # buf2 = buf2 * mag / 255
                cv2.multiply(self.vis_buf_u8_2, mag, scale=1.0/255.0, dtype=cv2.CV_8U, dst=self.vis_buf_u8_2)
                
                # 4. Merge
                # Note: cv2.merge might not support dst in all bindings, but we try to use it if possible
                # or just let it allocate one array (much better than 8)
                # We use self.vis_rgb_buf as destination if possible, or just assign to it
                # In Python cv2.merge returns the array. We can't easily force it to use existing buffer 
                # without using mixChannels or similar, which is complex.
                # So we accept one allocation here for 'rgb'
                rgb = cv2.merge([self.vis_buf_u8_1, self.vis_buf_u8_2, mag])
                
                # Create image
                flow_img = image.from_bytes(self.width, self.height, image.Format.FMT_RGB888, rgb.tobytes())
            else:
                # Fallback to numpy
                mag = np.abs(flow[0]) + np.abs(flow[1])
                mag_norm = np.clip(mag / 255.0, 0, 1)
                
                fx_scaled = (flow[0] * 3 + 127) * mag_norm
                fy_scaled = (flow[1] * 3 + 127) * mag_norm
                
                rgb_float = np.stack([fx_scaled, fy_scaled, mag], axis=-1)
                rgb = np.clip(rgb_float, 0, 255).astype(np.uint8)
                
                flow_img = image.from_bytes(self.width, self.height, image.Format.FMT_RGB888, rgb.tobytes())
            
            return flow_img
        except Exception as e:
            print(f"Error in _visualize_flow: {e}")
            return img

    
    def warmup(self):
        """Run warmup iterations."""
        print(f"\nWarmup ({self.warmup_times} iterations)...")
        
        frame1 = self.cam.read()
        self._preprocess(frame1, self.frame1_np)
        
        for i in range(self.warmup_times):
            frame2 = self.cam.read()
            self._preprocess(frame2, self.frame2_np)
            self._inference()
            self.frame1_np, self.frame2_np = self.frame2_np, self.frame1_np
            
            if (i + 1) % 5 == 0:
                print(f"  {i + 1}/{self.warmup_times}")
        
        print("✓ Warmup complete")
    
    def benchmark(self):
        """Run benchmark and return timing statistics."""
        if self.visualize:
            print(f"\nVisualization mode (running until Ctrl+C)...")
        else:
            print(f"\nBenchmark ({self.benchmark_times} iterations)...")
        
        times_camera = []
        times_preprocess = []
        times_inference = []
        times_visualization = []
        
        i = 0
        try:
            while True:
                # В режимі візуалізації - безкінечний цикл, інакше - фіксована кількість
                if not self.visualize and i >= self.benchmark_times:
                    break
                
                t_start = time.ticks_us()
                frame2 = self.cam.read()
                t_cam = time.ticks_us() - t_start
                
                t_start = time.ticks_us()
                self._preprocess(frame2, self.frame2_np)
                t_prep = time.ticks_us() - t_start
                
                t_start = time.ticks_us()
                self._inference()
                t_inf = time.ticks_us() - t_start
                
                # Візуалізація якщо увімкнена
                t_vis = 0
                if self.visualize:
                    # Ініціалізуємо дисплей при першій візуалізації
                    if self.disp is None:
                        try:
                            print("\nInitializing display...")
                            self.disp = display.Display()
                            print("✓ Display initialized")
                        except Exception as e:
                            print(f"✗ Display initialization failed: {e}")
                            self.visualize = False
                    
                    if self.disp is not None:
                        t_start = time.ticks_us()
                        try:
                            t1 = time.ticks_us()
                            flow = self._get_flow()
                            t2 = time.ticks_us()
                            flow_img = self._visualize_flow(frame2, flow)
                            t3 = time.ticks_us()
                            self.disp.show(flow_img)
                            t4 = time.ticks_us()
                            
                            if i == 0 or (i + 1) % 50 == 0:
                                print(f"  Vis breakdown: get_flow={((t2-t1)/1000):.1f}ms "
                                      f"visualize={((t3-t2)/1000):.1f}ms "
                                      f"show={((t4-t3)/1000):.1f}ms")
                        except Exception as e:
                            if i == 0:
                                print(f"Warning: Visualization error: {e}")
                            self.visualize = False
                        t_vis = time.ticks_us() - t_start
                
                times_camera.append(t_cam)
                times_preprocess.append(t_prep)
                times_inference.append(t_inf)
                times_visualization.append(t_vis)
                
                self.frame1_np, self.frame2_np = self.frame2_np, self.frame1_np
                
                if (i + 1) % 10 == 0:
                    avg_cam = sum(times_camera[-10:]) / 10 / 1000
                    avg_prep = sum(times_preprocess[-10:]) / 10 / 1000
                    avg_inf = sum(times_inference[-10:]) / 10 / 1000
                    avg_vis = sum(times_visualization[-10:]) / 10 / 1000
                    avg_total = avg_cam + avg_prep + avg_inf + avg_vis
                    fps = 1000 / avg_total
                    if self.visualize:
                        print(f"  {i+1}: Cam={avg_cam:.2f}ms Prep={avg_prep:.2f}ms "
                              f"Inf={avg_inf:.2f}ms Vis={avg_vis:.2f}ms Total={avg_total:.2f}ms FPS={fps:.1f}")
                    else:
                        print(f"  {i+1}/{self.benchmark_times}: Cam={avg_cam:.2f}ms Prep={avg_prep:.2f}ms "
                              f"Inf={avg_inf:.2f}ms Total={avg_total:.2f}ms FPS={fps:.1f}")
                
                i += 1
        except KeyboardInterrupt:
            if self.visualize:
                print(f"\n\n✓ Stopped after {i} iterations")
            else:
                raise
        
        return {
            'camera': times_camera,
            'preprocess': times_preprocess,
            'inference': times_inference,
            'visualization': times_visualization
        }
    
    def print_results(self, times_dict):
        """Print benchmark results.
        
        Args:
            times_dict: Dictionary with timing lists
        """
        times_camera = times_dict['camera']
        times_preprocess = times_dict['preprocess']
        times_inference = times_dict['inference']
        times_visualization = times_dict.get('visualization', [0] * len(times_camera))
        
        avg_cam = sum(times_camera) / len(times_camera) / 1000
        avg_prep = sum(times_preprocess) / len(times_preprocess) / 1000
        avg_inf = sum(times_inference) / len(times_inference) / 1000
        avg_vis = sum(times_visualization) / len(times_visualization) / 1000
        avg_total = avg_cam + avg_prep + avg_inf + avg_vis
        fps = 1000 / avg_total
        
        print(f"\n{'='*60}")
        print("BENCHMARK RESULTS")
        print(f"{'='*60}")
        print(f"Model:         {os.path.basename(self.model_path)}")
        print(f"Size:          {self.width}x{self.height}")
        print(f"Fuse:          {'Yes' if self.has_fuse else 'No'}")
        print(f"Preprocess:    {self.preprocess_method}")
        print(f"Visualize:     {'Yes' if self.visualize else 'No'}")
        print(f"Iterations:    {len(times_camera)}")
        print(f"{'-'*60}")
        print(f"Camera:        {avg_cam:>8.2f} ms  ({avg_cam/avg_total*100:5.1f}%)")
        print(f"Preprocessing: {avg_prep:>8.2f} ms  ({avg_prep/avg_total*100:5.1f}%)")
        print(f"Inference:     {avg_inf:>8.2f} ms  ({avg_inf/avg_total*100:5.1f}%)")
        if self.visualize:
            print(f"Visualization: {avg_vis:>8.2f} ms  ({avg_vis/avg_total*100:5.1f}%)")
        print(f"{'-'*60}")
        print(f"Total:         {avg_total:>8.2f} ms")
        print(f"FPS:           {fps:>8.1f}")
        print(f"{'='*60}")
    
    def cleanup(self):
        """Cleanup model resources."""
        print("\nCleaning up...")
        self.lib.CVI_NN_CleanupModel(self.model_handle)
        print("✓ Cleanup complete")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("\nUsage: python maixcam_optical_flow_benchmark.py <model.cvimodel> [--visualize]")
        print("\nOptions:")
        print("  --visualize    Enable optical flow visualization on display")
        print("\nExample:")
        print("  python maixcam_optical_flow_benchmark.py model.cvimodel")
        print("  python maixcam_optical_flow_benchmark.py model.cvimodel --visualize")
        return 1
    
    model_path = sys.argv[1]
    visualize = '--visualize' in sys.argv or '-v' in sys.argv
    
    if not os.path.exists(model_path):
        print(f"\n✗ Error: Model not found: {model_path}")
        return 1
    
    try:
        bench = OpticalFlowBenchmark(model_path, warmup=10, benchmark=100, visualize=visualize)
        bench.warmup()
        times = bench.benchmark()
        bench.print_results(times)
        bench.cleanup()
        
        print("\n✓ Benchmark completed successfully")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n✓ Interrupted by user")
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
