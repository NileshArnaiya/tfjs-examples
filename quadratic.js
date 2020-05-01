// TensorPlayground.com
// RGB Tensor
// INPUT TENSOR SHAPE: [622,1024,3]

(tf, aTensor) => {
  // return tensor to show
  const halfWidth = aTensor.shape[1]/2;
  const halfHeight = Math.round(aTensor.shape[0]/3.2);
  
  console.log(halfWidth);
  console.log(halfHeight);
  
  
  const halfCropped = tf.slice3d(aTensor, 250, [halfWidth, halfHeight]);
  
  const reverse = halfCropped.reverse(1);
  
  const combinedImg = tf.concat([halfCropped, reverse],1)
  
  return combinedImg 
  
}
  
  
  
  
  
  
  
  
  
  
