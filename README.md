# VIdeo-Enhancement

## TASK

The primary goal of this assignment is to showcase your proficiency in developing an
advanced AI model capable of enhancing the quality of a video by upscaling its
resolution and reducing noise. Your task entails harnessing the capabilities of any
suitable and available model to achieve significant improvements in video quality,
particularly by increasing its resolution and eliminating unwanted noise.
Essentially, we want you to demonstrate something that imitates this tool in video
enhancement: https://tensorpix.ai/
(Example: The input video will contain a lip-synced video of a person speaking in a
different language, since this is generative video, the area around the lips is pixelated
and contains noise. Upscale it, so the blur and pixelation goes away)


# Using Video Super Resolution with OpenVINO™
Super Resolution is the process of enhancing the quality of an image by increasing the pixel count using deep learning. This notebook applies Single Image Super Resolution (SISR) to frames in a 360p (480×360) video in 360p resolution. A model called [single-image-super-resolution-1032](https://docs.openvino.ai/2023.0/omz_models_model_single_image_super_resolution_1032.html), which is available in Open Model Zoo, is used in this tutorial. It is based on the research paper cited below.

> **NOTE**: The Single Image Super Resolution (SISR) model used in this demo is not optimized for a video. Results may vary depending on the video.

#### Table of contents:
- [Preparation](#Preparation)
    - [Install requirements](#Install-requirements)
```
pip install -q "openvino>=2023.1.0"
pip install -q opencv-python
pip install -q "pytube>=12.1.0"
pip install ipywidgets
```
Download the model from https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/{model_name}/FP16/single-image-super-resolution-1032
## Load the Superresolution Model
[back to top ⬆️](#Table-of-contents:)

Load the model in OpenVINO Runtime with `core.read_model` and compile it for the specified device with `core.compile_model`.
```
core = ov.Core()
model = core.read_model(model=model_xml_path)
compiled_model = core.compile_model(model=model, device_name=device.value)
```

### Do Inference
[back to top ⬆️](#Table-of-contents:)

Read video frames and enhance them with superresolution. Save the superresolution video, the bicubic video and the comparison video to a file.

The code below reads the video frame by frame. Each frame is resized and reshaped to the network input shape and upsampled with bicubic interpolation to the target shape. Both the original and the bicubic images are propagated through the network. The network result is a numpy array with floating point values, with a shape of `(1,3,1920,1080)`. This array is converted to an 8-bit image with the `(1080,1920,3)` shape and written to a `superres_video`. The bicubic image is written to a `bicubic_video` for comparison. Finally, the bicubic and result frames are combined side by side and written to a `comparison_video`. A progress bar shows the progress of the process. Both inference time and total time to process each frame are measured. That also includes inference time as well as the time it takes to process and write the video.
```
start_time = time.perf_counter()
frame_nr = 0
total_inference_duration = 0

progress_bar = ProgressBar(total=total_frames)
progress_bar.display()

cap = cv2.VideoCapture(filename=str(video_path))
try:
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            cap.release()
            break

        if frame_nr >= total_frames:
            break

        # Resize the input image to the network shape and convert it from (H,W,C) to
        # (N,C,H,W).
        resized_image = cv2.resize(src=image, dsize=(input_width, input_height))
        input_image_original = np.expand_dims(resized_image.transpose(2, 0, 1), axis=0)

        # Resize and reshape the image to the target shape with bicubic
        # interpolation.
        bicubic_image = cv2.resize(
            src=image, dsize=(target_width, target_height), interpolation=cv2.INTER_CUBIC
        )
        input_image_bicubic = np.expand_dims(bicubic_image.transpose(2, 0, 1), axis=0)

        # Do inference.
        inference_start_time = time.perf_counter()
        result = compiled_model(
            {
                original_image_key.any_name: input_image_original,
                bicubic_image_key.any_name: input_image_bicubic,
            }
        )[output_key]
        inference_stop_time = time.perf_counter()
        inference_duration = inference_stop_time - inference_start_time
        total_inference_duration += inference_duration

        # Transform the inference result into an image.
        result_frame = convert_result_to_image(result=result)

        # Write the result image and the bicubic image to a video file.
        superres_video.write(image=result_frame)
        bicubic_video.write(image=bicubic_image)

        stacked_frame = np.hstack((bicubic_image, result_frame))
        comparison_video.write(image=stacked_frame)

        frame_nr = frame_nr + 1

        # Update the progress bar and the status message.
        progress_bar.progress = frame_nr
        progress_bar.update()
        if frame_nr % 10 == 0 or frame_nr == total_frames:
            clear_output(wait=True)
            progress_bar.display()
            display(
                Pretty(
                    f"Processed frame {frame_nr}. Inference time: "
                    f"{inference_duration:.2f} seconds "
                    f"({1/inference_duration:.2f} FPS)"
                )
            )


except KeyboardInterrupt:
    print("Processing interrupted.")
finally:
    superres_video.release()
    bicubic_video.release()
    comparison_video.release()
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"Video's saved to {comparison_video_path.parent} directory.")
    print(
        f"Processed {frame_nr} frames in {duration:.2f} seconds. Total FPS "
        f"(including video processing): {frame_nr/duration:.2f}. "
        f"Inference FPS: {frame_nr/total_inference_duration:.2f}."
    )
```

Sample Result [Link](https://drive.google.com/file/d/1o3sKWFU349h8Uddlwg8qmQUpKjLLASsr/view?usp=drive_link)
