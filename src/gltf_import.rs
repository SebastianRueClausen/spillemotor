use anyhow::Result;
use glam::Mat4;

use std::path::{Path, PathBuf};
use std::{fs, io};

use crate::scene::{ImageData, MaterialData, MeshData, SceneData};

struct Importer {
    buffer_data: Vec<Box<[u8]>>,
    parent_path: PathBuf,
    gltf: gltf::Gltf,
}

impl Importer {
    fn new(path: &Path) -> Result<Self> {
        let file = match fs::File::open(path) {
            Ok(file) => file,
            Err(err) => {
                return Err(anyhow!("can't read file {path:?}: {err}"));
            }
        };

        let reader = io::BufReader::new(file);
        let gltf = gltf::Gltf::from_reader(reader)?;

        let parent_path = path
            .parent()
            .expect("`path` doesn't have a parent directory")
            .to_path_buf();

        let buffer_data: Result<Vec<_>> = gltf
            .buffers()
            .map(|buffer| {
                Ok(match buffer.source() {
                    gltf::buffer::Source::Bin => gltf
                        .blob
                        .as_ref()
                        .map(|blob| blob.clone().into_boxed_slice())
                        .ok_or_else(|| anyhow!("no binary blob in gltf scene"))?,
                    gltf::buffer::Source::Uri(uri) => {
                        let path: PathBuf = [parent_path.as_path(), Path::new(uri)].iter().collect();
                        fs::read(&path)
                            .map_err(|err| anyhow!("can't read file {path:?}: {err}"))?
                            .into_boxed_slice()
                    }
                })
            })
            .collect();

        Ok(Self {
            gltf,
            buffer_data: buffer_data?,
            parent_path,
        })
    }

    fn get_buffer_data(
        &self,
        view: &gltf::buffer::View,
        offset: usize,
        size: Option<usize>,
    ) -> Vec<u8> {
        let start = view.offset() + offset;
        let end = view.offset() + offset + size.unwrap_or(view.length() - offset);
        self.buffer_data[view.buffer().index()][start..end].to_vec()
    }

    fn load_image_data(&self, source: &gltf::image::Source) -> Result<ImageData> {
        match source {
            gltf::image::Source::View { view, mime_type } => {
                let format = match *mime_type {
                    "image/png" => image::ImageFormat::Png,
                    "image/jpeg" => image::ImageFormat::Jpeg,
                    _ => return Err(anyhow!("image must be either png of jpeg")),
                };

                let input = self.get_buffer_data(view, 0, None);

                let image = image::load(io::Cursor::new(&input), format)?.into_rgba8();
                let data = image.as_raw().to_vec();

                Ok(ImageData {
                    data,
                    width: image.width(),
                    height: image.height(),
                })
            }
            gltf::image::Source::Uri { uri, .. } => {
                let uri = Path::new(uri);

                let path: PathBuf = [self.parent_path.as_path(), Path::new(uri)]
                    .iter()
                    .collect();

                let image = image::open(&path)?.into_rgba8();
                let data = image.as_raw().to_vec();

                Ok(ImageData {
                    data,
                    width: image.width(),
                    height: image.height(),
                })
            }
        }
    }

    fn load_scene_data(self) -> Result<SceneData> {
        let materials: Result<Vec<_>> = self.gltf
            .materials()
            .map(|mat| {
                let base_color = mat
                    .pbr_metallic_roughness()
                    .base_color_texture()
                    .ok_or_else(|| anyhow!("no base color texture on material"))?
                    .texture()
                    .source()
                    .source();
                let metallic = mat
                    .pbr_metallic_roughness()
                    .metallic_factor();
                let roughness = mat
                    .pbr_metallic_roughness()
                    .roughness_factor();
                Ok(MaterialData {
                    base_color: self.load_image_data(&base_color)?,
                    metallic,
                    roughness,
                })
            })
            .collect();

        let materials = materials?;

        let meshes: Result<Vec<_>> = self.gltf.nodes()
            .filter(|node| node.mesh().is_some())
            .map(|node| {
                let mesh = node.mesh().unwrap();
                let transform = node.transform().matrix();

                let mut meshes = Vec::default();

                for primitive in mesh.primitives() {
                    let Some(material) = primitive.material().index() else {
                        return Err(anyhow!("primitive {} doesn't have a material", primitive.index()));
                    };

                    let verts = {
                        let positions = primitive.get(&gltf::Semantic::Positions);
                        let normals = primitive.get(&gltf::Semantic::Normals);
                        let texcoords = primitive.get(&gltf::Semantic::TexCoords(0));

                        let (Some(positions), Some(texcoords), Some(normals)) =
                            (positions, texcoords, normals) else
                        {
                            return Err(anyhow!(
                                "prmitive {} doesn't have both positions, texcoords and normals",
                                primitive.index(),
                            ));
                        };

                        use gltf::accessor::{DataType, Dimensions};

                        fn check_format(
                            acc: &gltf::Accessor,
                            dt: DataType,
                            d: Dimensions,
                        ) -> Result<()> {
                            if dt != acc.data_type() {
                                return Err(anyhow!( "accessor {} must be a {dt:?}", acc.index()));
                            }
                            if d != acc.dimensions() {
                                return Err(anyhow!( "accessor {} must be a {d:?}", acc.index()));
                            }
                            Ok(())
                        }
                    
                        check_format(&positions, DataType::F32, Dimensions::Vec3)?;
                        check_format(&normals, DataType::F32, Dimensions::Vec3)?;
                        check_format(&texcoords, DataType::F32, Dimensions::Vec2)?;

                        let get_accessor_data = |acc: &gltf::Accessor| -> Result<Vec<u8>> {
                            let Some(view) = acc.view() else {
                                return Err(anyhow!("no view on accessor {}", acc.index()));
                            };
                            Ok(self.get_buffer_data(&view, acc.offset(), Some(acc.count() * acc.size())))
                        };

                        let positions = get_accessor_data(&positions)?;
                        let normals = get_accessor_data(&normals)?;
                        let texcoords = get_accessor_data(&texcoords)?;

                        assert_eq!(positions.len() / 3, texcoords.len() / 2);
                        assert_eq!(positions.len() / 3, normals.len() / 3);

                        positions
                            .as_slice()
                            .chunks(12)
                            .zip(normals.as_slice().chunks(12))
                            .zip(texcoords.as_slice().chunks(8))
                            .flat_map(|((p, n), c)| [
                                p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11],
                                n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7], n[8], n[9], n[10], n[11],
                                c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
                            ])
                            .collect()
                    };

                    let indices = {
                        let Some(indices) = primitive.indices() else {
                            return Err(anyhow!("primtive {} has no indices", primitive.index()));
                        };

                        let data_type = indices.data_type();
                        let dimensions = indices.dimensions();

                        use gltf::accessor::{DataType, Dimensions};
                        let (DataType::U16, Dimensions::Scalar) = (data_type, dimensions) else {
                            return Err(anyhow!(
                                "index attribute, accessor {}, must be scalar `u16`",
                                indices.index(),
                            ));
                        };

                        let Some(view) = indices.view() else {
                            return Err(anyhow!("no view on accessor {}", indices.index()));
                        };

                        let offset = indices.offset();
                        let size = indices.count() * indices.size();

                        self.get_buffer_data(&view, offset, Some(size))
                    };

                    meshes.push(MeshData {
                        verts,
                        indices,
                        material,
                        transform: Mat4::from_cols_array_2d(&transform),
                    })
                }

                Ok(meshes)
            })
            .collect();

        let meshes = meshes?.into_iter().flatten().collect();

        Ok(SceneData {
            meshes,
            materials,
        })
    }
}

pub fn load(path: &Path) -> Result<SceneData> {
    Importer::new(path)?.load_scene_data()
}