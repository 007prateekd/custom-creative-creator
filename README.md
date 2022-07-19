# custom-creative-creator

[![](https://badgen.net/badge/Docker/Pull%20Image/blue?icon=docker)](https://hub.docker.com/r/007prateekd/custom-creative-creator/)

<img src=docs/teaser.png width=700>

This is a web application which takes in your picture, a text, a font style and a creative (both from a set of predefined options) and generates a new creative with the text in the creative replaced by your text and the face in it replaced by your face.

## Features

- uploading a picture of your face (or capturing using webcam) and choosing a reference style image, it can stylize your face using that reference.
- writing some text of your choice and a choosing a reference font image, it can stylize the text into that font. Note that the text in the ad may have characters not present in the provided text. So that needs to be synthesized in the required style.
- choosing a creative and combining the face and text stylization from above, it can generate a modified creative with the modified face added and modified text positioned, scaled and given direction according to the user.
- the complete pipeline takes only ~25s per image. If using the same face style reference continously, the time reduces down to ~5s.

## Examples

- Face Stylization:    
  The following are the style reference, user's image and stylized image respectively.   
  <img src=docs/face-creative.png height=150> <img src=docs/face-original.png height=150> <img src=docs/face-stylized.png height=150>

- Text Stylization:  
  The following are the font image and the stylized text (for the input "RANDOM TEXT") respectively.   
  <img src=docs/text-creative.png height=60> <img src=docs/text-stylized.png height=60>

- Combination:   
  The following are the original creative and modified creative respectively.
  <img src=docs/creative-old.png width=500>
  <img src=docs/creative-new.png width=500>

## Acknowledgements

This code borrows from <a href="https://github.com/mchong6/JoJoGAN">JoJoGAN</a>, <a href="https://github.com/yuval-alaluf/restyle-encoder">ReStyle</a> and <a href="https://github.com/hologerry/AGIS-Net">AGIS-Net</a>. Some snippets from <a href="https://github.com/shaoanlu/face_toolbox_keras">Face-Toolbox-Keras</a>.
