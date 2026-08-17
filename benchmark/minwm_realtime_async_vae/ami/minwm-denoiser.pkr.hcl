packer {
  required_plugins {
    amazon = {
      source  = "github.com/hashicorp/amazon"
      version = "~> 1.3"
    }
  }
}

variable "region" {
  type    = string
  default = "us-east-2"
}

variable "source_ami_id" {
  type        = string
  description = "Reviewed EKS 1.35 AL2023 NVIDIA source AMI in the target region."
}

variable "subnet_id" {
  type        = string
  description = "Explicit temporary builder subnet; no legacy-cluster default is allowed."
}

variable "security_group_id" {
  type        = string
  description = "Explicit temporary builder security group."
}

variable "iam_instance_profile" {
  type        = string
  description = "Explicit least-privilege builder instance profile."
}

variable "image_reference" {
  type        = string
  description = "Digest-pinned leap-world/minwm-realtime denoiser image."

  validation {
    condition     = can(regex("@sha256:[0-9a-f]{64}$", var.image_reference))
    error_message = "The image_reference value must be pinned by a SHA256 digest."
  }
}

variable "source_git_sha" {
  type        = string
  description = "SGLang source commit used to build image_reference."

  validation {
    condition     = can(regex("^[0-9a-f]{40}$", var.source_git_sha))
    error_message = "The source_git_sha value must be a full Git commit SHA."
  }
}

variable "ami_name" {
  type        = string
  description = "Unique name for the baked AMI."
}

source "amazon-ebs" "minwm_denoiser" {
  region                = var.region
  source_ami            = var.source_ami_id
  instance_type         = "g6.2xlarge"
  subnet_id             = var.subnet_id
  security_group_id     = var.security_group_id
  iam_instance_profile  = var.iam_instance_profile
  communicator          = "ssh"
  ssh_interface         = "session_manager"
  ssh_username          = "ec2-user"
  ami_name              = var.ami_name
  ami_description       = "EKS 1.35 NVIDIA AMI with MinWM denoiser image layers preloaded"
  force_deregister      = false
  force_delete_snapshot = false

  launch_block_device_mappings {
    device_name           = "/dev/xvda"
    volume_type           = "gp3"
    volume_size           = 200
    iops                  = 16000
    throughput            = 1000
    delete_on_termination = true
    encrypted             = true
  }

  run_tags = {
    Name        = "${var.ami_name}-builder"
    project     = "world-model"
    environment = "image-bake"
    managed_by  = "packer"
    purpose     = "gpu-ami-bake"
  }

  tags = {
    Name           = var.ami_name
    project        = "world-model"
    environment    = "production"
    managed_by     = "packer"
    purpose        = "gpu-ami-bake"
    base_ami       = var.source_ami_id
    baked_image    = var.image_reference
    source_git_sha = var.source_git_sha
  }
}

build {
  sources = ["source.amazon-ebs.minwm_denoiser"]

  provisioner "shell" {
    script = "${path.root}/preload_container_image.sh"
    environment_vars = [
      "AWS_REGION=${var.region}",
      "IMAGE_REFERENCE=${var.image_reference}",
      "SOURCE_GIT_SHA=${var.source_git_sha}",
    ]
  }
}
