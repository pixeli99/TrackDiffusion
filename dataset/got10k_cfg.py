# dataset settings
train_pipeline = [
    dict(
        type='mmtrack.TransformBroadcaster',
        share_random_params=True,
        transforms=[
            dict(type='mmdet.LoadImageFromFile'),
            dict(
                type='mmtrack.LoadTrackAnnotations',
                with_instance_id=True,
                with_bbox=True),
            dict(type='mmdet.Resize', scale=(480, 320), keep_ratio=False),
        ]),
    dict(type='mmtrack.PackTrackInputs', ref_prefix='ref', num_key_frames=16)
]

train_dataloader = dict(
    dataset=dict(
        type="VISDataset",
        data_root='/',
        dataset_version='got10k',
        interval=2,
        ann_file='defaultShare/SA-1B/coco_format_json/got10k_train.json',
        data_prefix=dict(img_path='GOT10K/train'),
        pipeline=train_pipeline,
        load_as_video=True,
        ref_img_sampler=dict(
            num_ref_imgs=16,
            frame_range=100,
            filter_key_img=True,
            method='uniform')
    )
)