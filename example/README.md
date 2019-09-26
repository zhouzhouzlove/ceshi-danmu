# example

This document is about how to use face_veri. new annotation should be added when new new demo is added.

## identify_face
Identify_face is face verification demo. It has two steps , register and match.
Register is a process  saving face feature of input image into face database.
usage:
```shell
$ identify_face -m model_data -d face_base -i input_img 
```
Match is match face taken from camera video with face database. 
usage:
```shell
$ identify_face -m model_data -d face_base -i input_img -v 
```

## test_face_verify
test_face_verify is test demo which verify whether two image face are the same.
usage:
```shell
test_face_verify model_data input_img1 input_img2
```

