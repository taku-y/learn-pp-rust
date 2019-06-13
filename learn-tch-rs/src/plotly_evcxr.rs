pub fn setup_plotly() {
    let mut html = String::new();
    html.push_str(r#"
<script>
requirejs.config({
paths: { 
'plotly': ['//cdnjs.cloudflare.com/ajax/libs/plotly.js/1.33.1/plotly-basic.min'], 
},
});
</script>
}"#);
    println!("EVCXR_BEGIN_CONTENT text/html\n{}\nEVCXR_END_CONTENT", html);
}
