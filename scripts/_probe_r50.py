import timm
for name in ["resnet50", "resnet34"]:
    c = timm.get_pretrained_cfg(name)
    print(name, "tag=", c.tag if c else None)
    print("   url=", c.url if c else None)
m = timm.create_model("resnet50", pretrained=False, features_only=True,
                      out_indices=(0, 1, 2, 3, 4))
print("resnet50 channels=", m.feature_info.channels(),
      "reduction=", m.feature_info.reduction())
tags = timm.list_pretrained("resnet50*")
print("resnet50 pretrained tags (first 8):", tags[:8])
