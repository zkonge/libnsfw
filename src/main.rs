use std::{
    env::args,
    io,
    sync::{
        mpsc::{channel, Sender},
        Arc, Mutex,
    },
    thread::spawn,
};

use anyhow::Result;
use image::imageops::{self, FilterType};
use onnxruntime::{
    environment::Environment, ndarray::Array4, session::Session, tensor::OrtOwnedTensor,
    GraphOptimizationLevel,
};
use rouille::{self, input::post::BufferedFile, post_input, try_or_400, Response};
use serde::Serialize;

type TaskType = (Sender<Result<Prediction>>, Vec<u8>);

#[derive(Debug, Serialize)]
struct Prediction {
    drawings: f32,
    hentai: f32,
    neutral: f32,
    porn: f32,
    sexy: f32,
}

struct Predictor {
    worker_channel: Sender<TaskType>,
}

impl Predictor {
    fn new(model_path: String, worker_thread: u16) -> Self {
        let (tx, rx) = channel::<TaskType>();
        spawn(move || {
            let environment = Environment::builder().with_name("nsfw").build().unwrap();
            let mut session = environment
                .new_session_builder()
                .unwrap()
                .with_optimization_level(GraphOptimizationLevel::All)
                .unwrap()
                .with_number_threads(worker_thread as i16)
                .unwrap()
                .with_model_from_file(model_path)
                .expect("read model failed");
            for (result_tx, task) in rx.into_iter() {
                result_tx.send(real_predict(&mut session, task)).unwrap();
            }
        });
        Predictor { worker_channel: tx }
    }
    fn predict(&self, image_bytes: Vec<u8>) -> Result<Prediction> {
        let (tx, rx) = channel();
        self.worker_channel.send((tx, image_bytes)).unwrap();
        rx.recv().unwrap()
    }
}

fn real_predict(session: &mut Session, image_bytes: Vec<u8>) -> Result<Prediction> {
    let tensor = Array4::from_shape_vec((1, 224, 224, 3), image_bytes)?.mapv(|x| x as f32 / 255.);
    let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(vec![tensor])?;
    let output = outputs[0].view();
    let result = output.as_slice().unwrap();

    Ok(Prediction {
        drawings: result[0],
        hentai: result[1],
        neutral: result[2],
        porn: result[3],
        sexy: result[4],
    })
}

const HELPER: &str = r#"LibNSFW
Minimal HTTP server provides nsfw image detection.
more detail in https://github.com/zkonge/libnsfw

Usage:
    ./libnsfw bind_addr nsfw_model worker_thread
Example:
    ./libnsfw 127.0.0.1:8000 ./nsfw.onnx 4

HTTP Request:
    Just POST form-data with `image` key.
HTTP Response:
    {
        "drawings": 0.5251695,
        "hentai": 0.47225672,
        "neutral": 0.0011893457,
        "porn": 0.0011269405,
        "sexy": 0.00025754774
    }
"#;

fn main() -> Result<()> {
    let args = args();
    if args.len() < 4 {
        println!("{}", HELPER);
        return Ok(());
    }
    let mut args = args.skip(1);
    let bind_addr = args.next().expect("bind_addr not found");
    let model_path = args.next().expect("nsfw_model not found");
    let worker_thread: u16 = args
        .next()
        .expect("worker_thread not found")
        .parse()
        .expect("parse worker_thread failed");

    let predictor = Arc::new(Mutex::new(Predictor::new(model_path, worker_thread)));

    rouille::start_server(bind_addr, move |request| {
        let p = predictor.clone();
        rouille::log(request, io::stdout(), || {
            let post_input = try_or_400!(post_input!(request, { image: BufferedFile }));

            let img = try_or_400!(image::load_from_memory(&post_input.image.data));
            let resized_img = imageops::resize(&img.to_rgb8(), 224, 224, FilterType::Lanczos3);

            let prediction = p.lock().unwrap().predict(resized_img.to_vec());

            match prediction {
                Ok(r) => Response::json(&r),
                Err(e) => {
                    let json = try_or_400::ErrJson::from_err(e.root_cause());
                    Response::json(&json).with_status_code(400)
                }
            }
        })
    });
}
