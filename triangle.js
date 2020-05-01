// TensorPlayground.com
// No Input Tensor

(tf) => {
  // here's your inputs to calculate a triange's area
  const base = tf.tensor([20])
  const height = tf.tensor([12])
  
  // example of addition with tensors:
  const c = tf.add(height, base)
  c.print()
  
  const areaTri = tf.mul(base, height).mul(0.5)
  // Above values result in 32
  
  // now refactor the above math to calcuate the area of a triangle!
  // remember the formula is area = 1/2 * base * height
  // answer should be 120!
  return areaTri
}
