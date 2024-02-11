variable "ARCH" {
  default = "amd64"
}
variable "BASE_IMAGE" {
  default = null
}
variable "SLIM_BASE" {
  default = null
}

target "_build_args" {
  args = {
    BASE_IMAGE = BASE_IMAGE,
    SLIM_BASE = SLIM_BASE,
  }
  platforms = ["linux/${ARCH}"]
}

target wget {
  dockerfile = "docker/main/Dockerfile"
  target = "wget"
  inherits = ["_build_args"]
}

target deps {
  dockerfile = "docker/main/Dockerfile"
  target = "deps"
  inherits = ["_build_args"]
}

target rootfs {
  dockerfile = "docker/main/Dockerfile"
  target = "rootfs"
  inherits = ["_build_args"]
}

target wheels {
  dockerfile = "docker/main/Dockerfile"
  target = "wheels"
  inherits = ["_build_args"]
}

target devcontainer {
  dockerfile = "docker/main/Dockerfile"
  platforms = ["linux/amd64"]
  target = "devcontainer"
}

target frigate {
  dockerfile = "docker/main/Dockerfile"
  platforms = ["linux/amd64"]
  target = "frigate"
}

target "onnx-converter" {
  dockerfile = "docker/onnx/Dockerfile.base"
  context = "."
  contexts = {
    wget = "target:wget",
  }
  target = "onnx-converter"
}

target "onnx-wheels" {
  dockerfile = "docker/onnx/Dockerfile.base"
  context = "."
}

target "onnx-nvidia-wheels" {
  dockerfile = "docker/onnx/Dockerfile.nvidia"
  context = "."
}

target "onnx" {
  dockerfile = "docker/onnx/Dockerfile.base"
  context = "."
  contexts = {
    wget = "target:wget",
    frigate = "target:frigate",
    rootfs = "target:rootfs"
    wheels = "target:wheels"
  }
  platforms = ["linux/amd64"]
  target = "frigate-onnx"
  tags = ["frigate-onnx"]
}

target "devcontainer-onnx" {
  dockerfile = "docker/onnx/Dockerfile.base"
  context = "."
  contexts = {
    wget = "target:wget",
    devcontainer = "target:devcontainer"
  }
  platforms = ["linux/amd64"]
  target = "devcontainer-onnx"
  tags = ["frigate-devcontainer-onnx"]
}

target "onnx-nvidia" {
  dockerfile = "docker/onnx/Dockerfile.nvidia"
  context = "."
  contexts = {
    onnx-converter = "target:onnx-converter"
    frigate = "target:frigate"
    wheels = "target:wheels"
  }
  platforms = ["linux/amd64"]
  target = "frigate-nvidia-onnx"
  tags = ["frigate-nvidia-onnx"]
}

target "devcontainer-onnx-nvidia" {
  dockerfile = "docker/onnx/Dockerfile.nvidia"
  context = "."
  contexts = {
    onnx-converter = "target:onnx-converter"
    devcontainer = "target:devcontainer"
    wheels = "target:wheels"
  }
  platforms = ["linux/amd64"]
  target = "devcontainer-nvidia-onnx"
  tags = ["frigate-devcontainer-onnx-nvidia"]
}
