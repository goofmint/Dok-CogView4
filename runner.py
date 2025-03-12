import torch
from diffusers import CogView4Pipeline
import argparse
import boto3

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument(
    '--output',
    default='/opt/artifact',
    help='出力先ディレクトリを指定します。',
)
arg_parser.add_argument(
    '--prompt',
    default='',
    help='プロンプト',
)
arg_parser.add_argument(
    '--num',
    default=8,
    help='生成する画像の数',
)
arg_parser.add_argument(
    '--width',
    default=1024,
    help='生成する画像の幅',
)
arg_parser.add_argument(
    '--height',
    default=1024,
    help='生成する画像の高さ',
)

arg_parser.add_argument(
    '--id',
    default='',
    help='タスクIDを指定します。',
)
arg_parser.add_argument('--s3-bucket', help='S3のバケットを指定します。')
arg_parser.add_argument('--s3-endpoint', help='S3互換エンドポイントのURLを指定します。')
arg_parser.add_argument('--s3-secret', help='S3のシークレットアクセスキーを指定します。')
arg_parser.add_argument('--s3-token', help='S3のアクセスキーIDを指定します。')

args = arg_parser.parse_args()

s3 = None
if args.s3_token and args.s3_secret and args.s3_bucket:
    # S3クライアントの作成
    s3 = boto3.client(
        's3',
        endpoint_url=args.s3_endpoint if args.s3_endpoint else None,
        aws_access_key_id=args.s3_token,
        aws_secret_access_key=args.s3_secret)

output_path = f'{args.output}/output-{args.id}.png'

pipe = CogView4Pipeline.from_pretrained("THUDM/CogView4-6B", torch_dtype=torch.bfloat16).to("cuda")

images = pipe(
    prompt=args.prompt,
    height=int(args.height),
    width=int(args.width),
    guidance_scale=3.5,
    output_type="pil",
    num_images_per_prompt=int(args.num),
    num_inference_steps=50,
).images

for idx, img in enumerate(images):
    file_path = f'{args.output}/output-{args.id}-{idx}.png'
    img.save(file_path)
    if s3 is not None:
        s3.upload_file(
            Filename=file_path,
            Bucket=args.s3_bucket,
            Key=os.path.basename(file_path))
