use glam::{Vec2, UVec2};
use anyhow::Result;

use std::{fs, path::{Path, PathBuf}, io};

#[derive(serde::Deserialize)]
struct BmChar {
    #[serde(rename = "char")]
    codepoint: String,

    width: u32,
    height: u32,

    xoffset: i32,
    yoffset: i32,

    xadvance: i32,
    
    x: u32,
    y: u32,
}

#[derive(serde::Deserialize)]
struct BmInfo {
    size: u32,
}

#[derive(serde::Deserialize)]
struct BmFont {
    pages: Vec<String>,
    chars: Vec<BmChar>,
    info: BmInfo,
}

#[derive(Clone)]
pub struct Glyph {
    pub codepoint: char,

    pub scaled_dim: Vec2,
    pub scaled_offset: Vec2,

    pub dim: Vec2,
    pub pos: Vec2,

    pub texcoord_min: Vec2,
    pub texcoord_max: Vec2,

    pub advance: f32,
}

pub struct FontData {
    pub size: u32,
    pub atlas_dim: UVec2,
    pub atlas: Vec<u8>,
    pub glyphs: Vec<Glyph>,
}

impl FontData {
    pub fn new(metadata: &Path) -> Result<Self> {
        let file = match fs::File::open(metadata) {
            Ok(file) => file,
            Err(err) => {
                return Err(anyhow!("can't read file {metadata:?}: {err}"));
            }
        };

        let reader = io::BufReader::new(file);
        let font: BmFont = serde_json::from_reader(reader)?;

        let Some(atlas_name) = font.pages.iter().next() else {
            return Err(anyhow!("no pages in font"));
        };

        let parent_path = metadata
            .parent()
            .expect("`path` doesn't have a parent directory")
            .to_path_buf();

        let atlas_path: PathBuf = [parent_path.as_path(), &Path::new(&atlas_name)]
            .iter()
            .collect();

        let image = image::open(&atlas_path)?;
        let atlas_dim = UVec2::new(image.width(), image.height());
        let atlas_dimf = Vec2::new(atlas_dim.x as f32, atlas_dim.y as f32);
        let atlas = image.into_luma8().as_raw().to_vec();

        let size = font.info.size as f32;
        let glyphs: Result<Vec<_>> = font.chars
            .iter()
            .map(|c| {
                let Some(codepoint) = c.codepoint.chars().next() else {
                    return Err(anyhow!("empty char"));
                };

                let dim = Vec2::new(c.width as f32, c.height as f32);
                let pos = Vec2::new(c.x as f32, c.y as f32);

                let scaled_dim = dim / Vec2::splat(size);
                let scaled_offset = Vec2::new(c.xoffset as f32, c.yoffset as f32) / Vec2::splat(size);

                let dim = dim / atlas_dimf;
                let pos = pos / atlas_dimf;

                let texcoord_min = Vec2::new(pos.x, pos.y);
                let texcoord_max = texcoord_min + dim;

                Ok(Glyph {
                    codepoint,
                    dim,
                    pos,
                    texcoord_min,
                    texcoord_max,
                    scaled_dim,
                    scaled_offset,
                    advance: c.xadvance as f32 / size,
                })
            })
            .collect();

        let glyphs = glyphs?;  

        Ok(Self {
            size: font.info.size,
            atlas_dim,
            atlas,
            glyphs, 
        })
    }
}
