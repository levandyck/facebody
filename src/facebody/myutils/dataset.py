import os
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import pickle
from collections.abc import Mapping
import xml.etree.ElementTree as ET
from PIL import Image

# ----------------------------- Image Dataloader ----------------------------- #
class ImageDataLoader:
    """Load image data for feature extraction."""
    DEFAULT_TRANSFORM = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, root: str, batch_size: int=128, shuffle: bool=False,
                 num_workers: int=8, class_names: list=None, n_classes: int=None,
                 n_imgs_cat: int = None, transform=None, xml_dir: str=None,
                 apply_bbox_crop: bool=False,
    ):
        root = Path(root)
        if not root.is_dir():
            raise ValueError(f"'{root}' is not a valid directory")

        self.transform = transform or self.DEFAULT_TRANSFORM
        self.xml_dir = xml_dir
        self.apply_bbox_crop = apply_bbox_crop

        base_dataset = ImageFolder(root=str(root), transform=self.transform)

        if class_names:
            self._filter_class_names(base_dataset, class_names)
        elif n_classes is not None:
            self._filter_first_n_classes(base_dataset, n_classes)
        if n_imgs_cat is not None:
            self._limit_imgs_per_category(base_dataset, n_imgs_cat)

        if apply_bbox_crop:
            self.dataset = BboxCroppedImageDataset(
                image_folder_dataset=base_dataset,
                xml_dir=xml_dir,
                apply_crop=apply_bbox_crop,
                transform=self.transform
            )
        else:
            self.dataset = base_dataset

        self.class_to_idx = base_dataset.class_to_idx
        self.classes = base_dataset.classes

        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )

    def _filter_class_names(self, base_dataset, class_names: list, *,
                            strict: bool=True):
        """Only include specific classes, remapped to the requested order."""
        seen = set()
        requested = [c for c in class_names if not (c in seen or seen.add(c))]

        all_classes = base_dataset.classes
        missing = [c for c in requested if c not in all_classes]
        if missing and strict:
            raise ValueError(f"Requested class_names not found: {missing}")

        kept_classes = [c for c in requested if c in all_classes]
        if not kept_classes:
            raise ValueError("No valid class names remain after filtering; nothing to keep")

        # New mapping in requested order
        new_class_to_idx = {cls_name: idx for idx, cls_name in enumerate(kept_classes)}

        # Build filtered samples with remapped labels
        new_samples = []
        for img_path, orig_label in base_dataset.samples:
            orig_cls_name = all_classes[orig_label]
            if orig_cls_name in new_class_to_idx:
                new_label = new_class_to_idx[orig_cls_name]
                new_samples.append((img_path, new_label))

        # Overwrite dataset internals
        base_dataset.classes = kept_classes
        base_dataset.class_to_idx = new_class_to_idx
        base_dataset.samples = new_samples
        base_dataset.targets = [label for (_, label) in new_samples]
        if hasattr(base_dataset, "imgs"):
            base_dataset.imgs = list(new_samples)

    def _filter_first_n_classes(self, base_dataset, n: int):
        """Only include first n classes."""
        # Grab all classes and take first n
        all_classes = base_dataset.classes
        kept_classes = all_classes[:n]

        # Build new class mapping
        new_class_to_idx = {cls_name: idx for idx, cls_name in enumerate(kept_classes)}

        # Filter samples
        new_samples = []
        for img_path, orig_label in base_dataset.samples:
            orig_cls_name = all_classes[orig_label]
            if orig_cls_name in new_class_to_idx:
                new_label = new_class_to_idx[orig_cls_name]
                new_samples.append((img_path, new_label))

        # Overwrite
        base_dataset.classes = kept_classes
        base_dataset.class_to_idx = new_class_to_idx
        base_dataset.samples = new_samples
        base_dataset.targets = [label for (_, label) in new_samples]

    def _limit_imgs_per_category(self, base_dataset, n: int):
        """For each category, keep only first n images."""
        from collections import defaultdict
        # For each label, count # seen
        seen_per_class = defaultdict(int)
        new_samples = []
        for img_path, label in base_dataset.samples:
            if seen_per_class[label] < n:
                new_samples.append((img_path, label))
                seen_per_class[label] += 1
        base_dataset.samples = new_samples
        base_dataset.targets = [label for (_, label) in new_samples]
        if hasattr(base_dataset, "imgs"):
            base_dataset.imgs = list(new_samples)

    def _filter_to_filename_whitelist(self, per_class_files, *, per_class_limit=None,
                                      match_basename=True, strict=False):
        """Keep only images whose file names are whitelisted per class."""
        ds = self.dataset
        classes = ds.classes
        class_to_idx = ds.class_to_idx

        # Normalize whitelist to mapping: class_name -> ordered list of names to keep
        if isinstance(per_class_files, Mapping):
            requested = {str(k): list(v) for k, v in per_class_files.items()}
            unknown = [k for k in requested.keys() if k not in class_to_idx]
            if unknown and strict:
                raise ValueError(f"Unknown class names in whitelist: {unknown}")
        else:
            # Single list applied to every class
            requested = {c: list(per_class_files) for c in classes}

        # Optionally truncate to first N per class
        if per_class_limit is not None:
            for c in requested:
                requested[c] = requested[c][:per_class_limit]

        # Build lookup sets per class using basenames or full names
        keep_sets = {}
        for c, names in requested.items():
            base_names = [os.path.basename(n) for n in names]
            keep_sets[c] = set(base_names if match_basename else names)

        # Filter samples and remap labels
        new_samples = []
        for img_path, y in ds.samples:
            cls_name = classes[y]
            keep = keep_sets.get(cls_name, None)
            if not keep:
                continue
            fname = os.path.basename(img_path) if match_basename else img_path
            if fname in keep:
                new_samples.append((img_path, y))

        if strict:
            empty = [c for c in requested if c in class_to_idx and
                    not any(classes[y] == c for _, y in new_samples)]
            if empty:
                raise ValueError(f"No images kept for classes: {empty}")

        # Overwrite
        ds.samples = new_samples
        ds.targets = [y for _, y in new_samples]
        if hasattr(ds, "imgs"):
            ds.imgs = list(new_samples)

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)

    def save_class_info(self, out_path: str):
        """Save class_to_idx mapping."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump({"class_to_idx": self.class_to_idx,
                         "classes": self.classes
                         }, f)
        print(f"Saved class info to {out_path}")


# ----------------------- BoundingBox Image Dataloader ----------------------- #
class BboxCroppedImageDataset(Dataset):
    """Wrap ImageFolder dataset with optional bounding box cropping."""
    def __init__(self, image_folder_dataset: ImageFolder, 
                 xml_dir: str=None, 
                 apply_crop: bool=True,
                 transform=None, ids=None):
        self.image_folder_dataset = image_folder_dataset
        self.xml_dir = Path(xml_dir) if xml_dir else None
        self.apply_crop = apply_crop and (xml_dir is not None)
        self.transform = transform
        self.ids = list(ids) if ids is not None else None

    def _load_bbox_from_xml(self, xml_path):
        """Extract bounding box (xmin, ymin, xmax, ymax) from XML."""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            obj = root.find("object")
            if obj is None:
                return None
            bndbox = obj.find("bndbox")
            if bndbox is None:
                return None
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            return xmin, ymin, xmax, ymax
        except Exception as e:
            print(f"Warning: Failed to parse XML {xml_path}: {e}")
            return None

    def __len__(self):
        if self.ids is not None:
            return len(self.ids)
        return len(self.image_folder_dataset)

    def __getitem__(self, i):
        idx = self.ids[i] if self.ids is not None else i
        img_path, label = self.image_folder_dataset.samples[idx]
        img = Image.open(img_path).convert("RGB")

        # Apply bounding box crop if available
        if self.apply_crop:
            if self.xml_dir:
                # Stanford40: use XML bbox
                img_path_obj = Path(img_path)
                category = img_path_obj.parent.name
                filename = img_path_obj.stem + ".xml"
                xml_path = Path(self.xml_dir) / category / filename

                if xml_path.exists():
                    bbox = self._load_bbox_from_xml(xml_path)
                    if bbox:
                        xmin, ymin, xmax, ymax = bbox
                        xmin = max(0, int(xmin))
                        ymin = max(0, int(ymin))
                        xmax = min(img.width, int(xmax))
                        ymax = min(img.height, int(ymax))
                        if xmax > xmin and ymax > ymin:
                            img = img.crop((xmin, ymin, xmax, ymax))
                else:
                    print(f"XML file not found: {xml_path}")
            else:
                # Celeb-reID
                pass

            # Pad to square
            width, height = img.size
            if width != height:
                max_side = max(width, height)
                new_img = Image.new("RGB", (max_side, max_side), (128, 128, 128))

                # Center paste
                paste_x = (max_side - width) // 2
                paste_y = (max_side - height) // 2
                new_img.paste(img, (paste_x, paste_y))
                img = new_img

        # Apply transform if provided
        if self.transform:
            img = self.transform(img)

        return img, label
