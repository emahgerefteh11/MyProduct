variable "aws_region" {
  description = "AWS deployment region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Target deployment environment (dev, staging, prod)"
  type        = string
  default     = "default"
}

variable "platform" {
  description = "Platform / Organization unit"
  type        = string
  default     = "unknown"
}

variable "product" {
  description = "Product / Workload identifier"
  type        = string
  default     = "unknown"
}

variable "instance_type" {
  description = "EC2 instance size based on discovered billing SKU"
  type        = string
  default     = "c5.xlarge"
}

variable "ami_id" {
  description = "Custom AMI ID (leave empty to use default Amazon Linux 2)"
  type        = string
  default     = ""
}

variable "root_volume_size_gb" {
  description = "Root EBS volume size in GiB"
  type        = number
  default     = 30
}

variable "ebs_volume_size_gb" {
  description = "Additional EBS volume size in GiB"
  type        = number
  default     = 100
}

variable "custom_tags" {
  description = "Additional user-defined resource tags"
  type        = map(string)
  default     = {}
}
