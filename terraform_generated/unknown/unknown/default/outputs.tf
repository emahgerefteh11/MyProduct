output "platform" {
  description = "Platform identifier"
  value       = var.platform
}

output "product" {
  description = "Product identifier"
  value       = var.product
}

output "environment" {
  description = "Target environment"
  value       = var.environment
}

output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.unknown_app.id
}

output "instance_type" {
  description = "Provisioned instance SKU"
  value       = aws_instance.unknown_app.instance_type
}
