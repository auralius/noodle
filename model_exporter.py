import numpy as np

def number_to_letters(n):
    if not (n < 26 * 26):
        return "??"
    first = n // 26
    second = n % 26
    return chr(ord('a') + first) + chr(ord('a') + second)

def to_two_digit_string(n: int) -> str:
    return f"{n:02d}"

def format_c_array(data, indent="                            ", items_per_line=4):
    formatted_values = [f"{val:.6e}" for val in data]
    if not formatted_values:
        return ""

    output_lines = []
    simple_formatted_values = []
    for i in range(len(data)):
        val_str = f"{data[i]:.6e}"
        if i < len(data) - 1:
            val_str += ","
        simple_formatted_values.append(val_str)

    output_lines_simple = []
    for i in range(0, len(simple_formatted_values), items_per_line):
        output_lines_simple.append(indent + " ".join(simple_formatted_values[i:i+items_per_line]))

    return "\n".join(output_lines_simple)

def exporter(weights, dir):
  if not dir.endswith('/'):
    dir += '/'

  w_idx = 0
  b_idx = 0
  for k in range(len(weights)):
    w = weights[k]
    if len(w.shape) == 4: # convolution kernel
      w_idx += 1
      for i in range(w.shape[2]):
          for o in range(w.shape[3]):
              fn_txt = dir + 'w' + to_two_digit_string(w_idx)+ number_to_letters(i) + number_to_letters(o) + '.txt'
              print(fn_txt)
              np.savetxt(fn_txt, np.float32(w[:, :, i, o].flatten()), fmt='%.6e', newline='\n')

              fn_h = fn_txt.replace('.txt', '.h')
              var_name = 'w' + to_two_digit_string(w_idx) + number_to_letters(i) + number_to_letters(o)
              c_array_content = format_c_array(np.float32(w[:, :, i, o].flatten()))

              with open(fn_h, 'w') as f_h:
                  f_h.write("#pragma once\n\n")
                  f_h.write(f"static const float {var_name}[] = {{\n")
                  f_h.write(c_array_content)
                  f_h.write("\n}};;\n")
              print(fn_h)

    elif len(w.shape) == 2: # weight for dense network
      w_idx += 1
      fn_txt = dir + 'w' + to_two_digit_string(w_idx) + '.txt'
      print(fn_txt)
      np.savetxt(fn_txt, np.float32(w.transpose().flatten()), fmt='%.6e', newline='\n')

      fn_h = fn_txt.replace('.txt', '.h')
      var_name = 'w' + to_two_digit_string(w_idx)
      c_array_content = format_c_array(np.float32(w.transpose().flatten()))

      with open(fn_h, 'w') as f_h:
          f_h.write("#pragma once\n\n")
          f_h.write(f"static const float {var_name}[] = {{\n")
          f_h.write(c_array_content)
          f_h.write("\n}};;\n")
      print(fn_h)

    elif len(w.shape) == 1: # bias for convolutional and dense network
      b_idx += 1
      fn_txt = dir + 'b' + to_two_digit_string(b_idx) + '.txt'
      print(fn_txt)
      np.savetxt(fn_txt, np.float32(w.flatten()), fmt='%.6e', newline='\n')

      fn_h = fn_txt.replace('.txt', '.h')
      var_name = 'b' + to_two_digit_string(b_idx)
      c_array_content = format_c_array(np.float32(w.flatten()))

      with open(fn_h, 'w') as f_h:
          f_h.write("#pragma once\n\n")
          f_h.write(f"static const float {var_name}[] = {{\n")
          f_h.write(c_array_content)
          f_h.write("\n}};;\n")
      print(fn_h)
