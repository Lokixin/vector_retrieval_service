from io import BytesIO

from fastapi import HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

MAX_IMAGE_UPLOAD_SIZE_BYTES = 5 * 1024 * 1024


async def read_image_upload(uploaded_image: UploadFile) -> Image.Image:
    if not uploaded_image.content_type or not uploaded_image.content_type.startswith(
        "image/"
    ):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Uploaded file must be an image",
        )

    image_bytes = await uploaded_image.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image cannot be empty",
        )
    if len(image_bytes) > MAX_IMAGE_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Uploaded image is too large",
        )

    try:
        image = Image.open(BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid image",
        ) from exc

    return image.convert("RGB")
