import * as tf from "@tensorflow/tfjs";
// 1,000 Dog and 1,000 Cat pics
import * as DogsNCats from "dogs-n-cats";
import "./styles.css";

document.getElementById("app").innerHTML = '<canvas id="printCanvas" />';
const printCanvas = document.getElementById("printCanvas");
DogsNCats.load().then(async dnc => {
  const newStack = tf.tidy(() => 
    {
      const [dncTensor] = dnc.training.get(15);
      return tf.concat(tf.unstack(dncTensor),1);
    });
  await tf.browser.toPixels(newStack, printCanvas);
  newStack.dispose();
});

