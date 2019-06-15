extern crate ndarray;
use std::collections::HashMap;
use ndarray::{Array1, Array2};

pub enum Value {
    A1(Array1<f32>),
    A2(Array2<f32>),
    S(String)
}

impl Value {
    pub fn to_string(&self) -> String {
        match &self {
            Value::A1(v) => format!("{}", &v).replace("\n", ""),
            Value::A2(v) => format!("{}", &v).replace("\n", ""),
            Value::S(s) => s.clone()
        }
    }
}

pub struct PlotlyData(pub HashMap<String, Value>);

impl PlotlyData {
    pub fn to_string(&self) -> String {
        let mut data = String::new();
        data.push_str("{");
        for (key, value) in &self.0 {
            data.push_str(
                format!("{}: {}, ", key, value.to_string()).as_str()
            );
        }
        data.push_str("}");

        data
    }
}

pub fn plotly_plot(data: &PlotlyData) {
    let iframe_start = r#"
    <style>
    iframe {border:0;}
    </style>
    <iframe width="600" height="600" srcdoc="
        <script src='https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.6/d3.min.js'></script>
        <script src='https://code.jquery.com/jquery-2.1.4.min.js'></script>
        <script src='https://cdn.plot.ly/plotly-latest.min.js'></script>

        <div id=&quot;myDiv&quot; style=&quot;width:500px;height:500px;&quot;></div>

        <script>
    "#;
    let iframe_end = r#"
    Plotly.plot('myDiv', data);
        </script>"
    </iframe>
    "#;
    let data_start = String::from(r#"var data = ["#);
    let data_end = r#"];"#;
    let mut html = String::new();
    let data_start = data_start.to_string();
    let data = data_start + &data.to_string() + &data_end;

    html.push_str(&iframe_start);
    html.push_str(&data);
    html.push_str(&iframe_end);
    // println!("{}", &html);
    println!("EVCXR_BEGIN_CONTENT text/html\n{}\nEVCXR_END_CONTENT", html);
}
