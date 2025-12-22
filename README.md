This is a personal project I am working on. The intent is to be an all-in-one place for infrastructure: analyzing existing infrastructure, breaking down cost by platform/product, categorizing SKUs per platform/product, providing FinOps views, and eventually producing Terraform to provision future builds. Currently it’s AWS-only with mock data.

## Release Roadmap (draft)
- **AWS (current):** CUR ingestion, FinOps overview, platform/product filters, resource naming breakdown, anomalies/distributions (EC2 SKUs, EBS/S3 buckets).
- **Upcoming:** 
  - GCP & Azure analysis parity (billing ingestion, SKU mapping, platform/product breakdowns).
  - Terraform buildouts: generate TF skeletons for discovered stacks (networking, compute, storage) per platform/product.
  - Resource naming convention inputs: input resource naming conventions related to products/platforms to produce reports by product/platform.
  - Resource naming convention analysis: determining resource naming conventions as they relate to platforms/products without inputs.

AWS data in this repo is mock and AI-generated. 
