import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="mini_rnn_float32.tflite")
interpreter.allocate_tensors()

print("=== 输入详情 ===")
print(interpreter.get_input_details())
print("\n=== 输出详情 ===")
print(interpreter.get_output_details())
