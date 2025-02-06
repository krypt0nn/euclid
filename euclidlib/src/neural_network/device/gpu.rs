use wgpu::util::DeviceExt;

use crate::prelude::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Perform computations on GPU.
pub struct GPUDevice {
    device: wgpu::Device,
    queue: wgpu::Queue,

    layer_forward_propagation_shader: wgpu::ShaderModule
}

impl GPUDevice {
    /// Try searching for GPU computations device with best performance.
    pub fn new() -> Option<Self> {
        let instance = wgpu::Instance::default();
        let adapter = smol::block_on(instance.request_adapter(&wgpu::RequestAdapterOptionsBase::default()))?;

        let capabilities = adapter.get_downlevel_capabilities();

        if !capabilities.flags.contains(wgpu::DownlevelFlags::COMPUTE_SHADERS) {
            return None;
        }

        let (device, queue) = smol::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::Performance
            },
            None
        )).ok()?;

        let layer_forward_propagation_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl("
                @group(0) @binding(0) var<storage, read> inputs: array<f32>;
                @group(0) @binding(1) var<storage, read> weights: array<f32>;
                @group(0) @binding(3) var<storage, read_write> outputs: array<f32>;

                @compute @workgroup_size(256, 256, 2)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let neuron_id = global_id.x;
                    let neurons_num = arrayLength(&outputs);

                    if (neuron_id >= neurons_num) {
                        return;
                    }

                    let input_id = global_id.y;
                    let inputs_num = arrayLength(&inputs);

                    if (input_id >= inputs_num) {
                        return;
                    }

                    let neuron_weights_offset = neuron_id * inputs_num;

                    if inputs_num <= 256 {
                        outputs[neuron_id] += inputs[input_id] * weights[neuron_weights_offset + input_id];
                    }

                    else {
                        let weights_per_thread = udiv(inputs_num, 256);
                        let weights_in_workgroup = weights_per_thread * 256;

                        if global_id.z == 0 {
                            let out_of_bounds_weights = inputs_num - weights_in_workgroup;

                            for (var i = 0; i < out_of_bounds_weights; i++) {
                                outputs[neuron_id] += inputs[weights_in_workgroup + i] * weights[neuron_weights_offset + weights_in_workgroup + i];
                            }
                        }

                        else {
                            let neuron_inputs_offset = input_id * weights_per_thread;

                            for (var i = 0; i < weights_per_thread; i++) {
                                let j = neuron_inputs_offset + i;

                                outputs[neuron_id] += inputs[j] * weights[neuron_weights_offset + j];
                            }
                        }
                    }
                }
            ".into())
        });

        Some(Self {
            device,
            queue,

            layer_forward_propagation_shader
        })
    }
}

impl Device for GPUDevice {
    unsafe fn forward<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize, F: Float>(
        &self,
        neurons: &[Neuron<INPUT_SIZE, F>; OUTPUT_SIZE],
        inputs: &[F; INPUT_SIZE],
        outputs: &mut [F; OUTPUT_SIZE]
    ) {
        // Prepare inputs bytes.
        let inputs_buf = inputs.iter()
            .map(F::as_f32)
            .collect::<Vec<f32>>();

        let inputs_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&inputs_buf),
            usage: wgpu::BufferUsages::STORAGE
        });

        // Prepare weights bytes.
        let weights_buf = neurons.iter()
            .flat_map(|neuron| neuron.weights.iter())
            .map(F::as_f32)
            .collect::<Vec<f32>>();

        let weights_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&weights_buf),
            usage: wgpu::BufferUsages::STORAGE
        });

        // Prepare outputs bytes (biases of neurons).
        let outputs_buf = neurons.iter()
            .map(|neuron| neuron.bias.as_f32())
            .collect::<Vec<f32>>();

        let outputs_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&outputs_buf),
            usage: wgpu::BufferUsages::STORAGE
        });

        // Prepare GPU buffer types layout.
        let bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                // Inputs buffer.
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: Some(std::num::NonZeroU64::new_unchecked(4)),
                        has_dynamic_offset: false
                    },
                    count: std::num::NonZeroU32::new(inputs.len() as u32)
                },

                // Weights buffer.
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: Some(std::num::NonZeroU64::new_unchecked(4)),
                        has_dynamic_offset: false
                    },
                    count: std::num::NonZeroU32::new((neurons.len() * inputs.len()) as u32)
                },

                // Outputs buffer.
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        min_binding_size: Some(std::num::NonZeroU64::new_unchecked(4)),
                        has_dynamic_offset: false
                    },
                    count: std::num::NonZeroU32::new(neurons.len() as u32)
                }
            ]
        });

        // Prepare bindings from layout and buffers.
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: inputs_buf.as_entire_binding()
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weights_buf.as_entire_binding()
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: outputs_buf.as_entire_binding()
                }
            ]
        });

        // Prepare pipeline layout.
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[]
        });

        // Create pipeline from the layout and compiled shader.
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &self.layer_forward_propagation_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None
        });

        // Create commands encoder.
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // Prepare command to calculate weighted outputs of neurons.
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None
        });

        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(256, 256, 2);

        drop(compute_pass);

        let command_buffer = encoder.finish();

        // Submit this command to the GPU.
        self.queue.submit([command_buffer]);

        // Wait for GPU to finish the command and read calculated outputs.
        self.device.poll(wgpu::Maintain::Wait);

        let outputs_buf = outputs_buf.slice(..).get_mapped_range();
        let outputs_buf = bytemuck::cast_slice::<_, f32>(&outputs_buf);

        for (i, output) in outputs_buf.iter().enumerate() {
            outputs[i] = (neurons[i].activation_function)(F::from_float(*output));
        }
    }
}

#[test]
fn test_gpu_forward() {
    if let Some(gpu) = GPUDevice::new() {
        let mut cpu_outputs = [0.0];
        let mut gpu_outputs = [0.0];

        let neurons = [
            Neuron32::default()
                .with_activation_function(linear, linear_derivative)
                .with_weights([0.5, 1.0])
                .with_bias(2.0)
        ];

        unsafe {
            CPUDevice::default().forward(&neurons, &[0.4, 0.5], &mut cpu_outputs);
            gpu.forward(&neurons, &[0.4, 0.5], &mut gpu_outputs);
        }

        assert_eq!(cpu_outputs, [2.7]);
        assert_eq!(cpu_outputs, gpu_outputs);
    }
}
