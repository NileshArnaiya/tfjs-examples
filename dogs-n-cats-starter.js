import * as tf from "@tensorflow/tfjs";
import "./styles.css";
import * as DogsNCats from "dogs-n-cats";

document.getElementById("app").innerHTML = `
<h1>It's Training Dogs n Cats</h1>
<button id='load'>Load Dogs n Cats</button>
<button id='create'>Create Model</button>
<button id='train'>Train Model</button>
<button id='test'>Test Model</button>
<button id='dispose'>Dispose Model</button>
`;

let dnc, model;
document.getElementById("load").onclick = async () => {
  console.log("Loading Dogs N Cats Data");
  dnc = await DogsNCats.load();
};

document.getElementById("create").onclick = async () => {
  console.log("Creating our Model");
  model = tf.sequential();
  model.add(
    tf.layers.conv2d({
        inputShape:[32,32,3],
        kernelSize: 3,
        padding: "same",
        filters:64,
        strides:1,
        activation:"relu",
        kernelInitializer:"heNormal"
      })
  );

  model.add(
    tf.layers.maxPool2d({
      poolSize:[2,2],
      strides:[2,2],      
      })
  );

  model.add(
    tf.layers.batchNormalization());

  model.add(tf.layers.dropout({
    rate: 0.25
  })
  );
  
    model.add(tf.layers.flatten());

};

document.getElementById("train").onclick = async () => {
  console.log("Training the model");
};

document.getElementById("test").onclick = async () => {
  console.log("Testing Model");
};

document.getElementById("dispose").onclick = async () => {
  console.log("Cleaning up!");
};


