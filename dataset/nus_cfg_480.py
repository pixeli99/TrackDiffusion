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

test_pipeline = [
    dict(type='mmtrack.LoadImageFromFile'),
    dict(
        type='mmtrack.LoadTrackAnnotations',
        with_instance_id=True,
        with_bbox=True),
    dict(type='mmdet.Resize', scale=(480, 320), keep_ratio=False),
    dict(type='mmtrack.PackTrackInputs', pack_single_img=True)
]


dataset_type = 'VISDataset'
data_root = '/'
# dataloader
train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        dataset_version='nus',
        ann_file='mmdetection3d/nuscenes_vis.json',
        data_prefix=dict(img_path=''),
        pipeline=train_pipeline,
        load_as_video=True,
        ref_img_sampler=dict(
            num_ref_imgs=16,
            frame_range=100,
            filter_key_img=True,
            method='uniform')))

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        dataset_version='nus',
        ann_file='mmdetection3d/nuscenes_vis_val.json',
        data_prefix=dict(img_path=''),
        pipeline=test_pipeline,
        test_mode=True,
        ref_img_sampler=None,
        load_as_video=True,)
)