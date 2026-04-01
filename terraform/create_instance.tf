# Fetch the latest Ubuntu 24.04 LTS AMI for x86_64
data "aws_ami" "ubuntu_24" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

resource "aws_instance" "ollama_gpu" {
  # Change this to "g6.2xlarge" for NVIDIA L4 GPUs
  instance_type = "g5.2xlarge" 
  ami           = data.aws_ami.ubuntu_24.id
  key_name      = "your-key-pair"

  root_block_device {
    volume_size = 100 
    volume_type = "gp3"
  }

  user_data = <<-EOF
              #!/bin/bash
              # Update and install NVIDIA drivers for Ubuntu 24.04
              sudo apt-get update -y
              sudo apt-get install -y nvidia-driver-535 nvidia-utils-535
              
              # Install Ollama
              curl -fsSL https://ollama.com | sh
              
              # Set critical context variables for large window (262k)
              echo "OLLAMA_CONTEXT_LENGTH=262144" >> /etc/environment
              
              # Optional: Pre-pull the model (caution: takes time on first boot)
              # ollama pull kavai/qwen3.5-GPT5:9b
              EOF

  tags = {
    Name = "Ollama-Ubuntu24-GPU"
  }
}
