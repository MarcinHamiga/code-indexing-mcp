variable "region" {
  default = "eu"
}

resource "aws_instance" "web" {
  ami = "ami-123"
}
