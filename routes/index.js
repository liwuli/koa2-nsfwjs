const router = require('koa-router')()
// 导入必要的模块
const axios = require('axios'); 
// 导入nsfwjs模块
const nsfwjs = require('nsfwjs');
// jpeg-js
const jpeg = require('jpeg-js')
var gifFrames = require('gif-frames');

const tf = require('@tensorflow/tfjs-node')
const fs = require('fs');
const path = require('path');
// 模型
let _model = null; 
tf.enableProdMode()

router.get('/', async (ctx, next) => {
  await ctx.render('index', {
    title: 'Hello Koa 2!'
  })
})

// 定义鉴黄路由
router.post('/nsfw', async (ctx, next) => {
  try {
    // 获取图片数据
    const fileData = await fs.promises.readFile(ctx.request.files.imageData.filepath);
    // 获取文件后缀
    const ext = path.extname(ctx.request.files.imageData.originalFilename).toLowerCase();
    // 判断imageData不为空
    if (fileData) {
      if(ext === '.png' || ext === '.jpeg' || ext === '.jpg'){
        // 文件对象buff转换成张量
        const image = await tf.node.decodeImage(fileData,3);    
        // 执行预测
        const predictions = await _model.classify(image)
        ctx.body = {
          predictions: predictions
        };
      }else if(ext === '.gif'){
        // 把gif的每帧抽离出来  然后进行识别
        let frameInfos = await gifFrames({url: fileData, frames: 'all',outputType: 'jpg',cumulative: true});        
        
        // 随机抽取帧里面的5个数据，如果里面的小于5就不抽取
        if (frameInfos.length >= 5) {
          frameInfos = frameInfos.sort(() => Math.random() - 0.5).slice(0, 5); 
        } 
        // 把多个预测弄成 promises
        const promises = frameInfos.map(async function (frame) {
          return new Promise(async function (resolve, reject) {
            try {
              // 获取当前帧的流对象
              const jpgStream = frame.getImage();            
              // 文件对象buff转换成张量
              const image =  await tf.node.decodeImage(jpgStream._obj,3);     
              //执行预测 
              const predictions = await _model.classify(image);
              // 返回结果
              resolve(predictions);
            } catch (error) {
              console.log('抽帧检测失败！',error);
              reject(false);
            }
          })          
        });
        //等待所有的数据返回 
        const predictions = await Promise.all(promises);     
        // const image = await tf.node.decodeGif(fileData);   
        // const predictions = await _model.classifyGif(image);
        // 返回给客户端
        ctx.body = {
          predictions: predictions
        };
      }else{
        ctx.status = 500;
        ctx.body = '目前只支持png和jpg格式！';
      }
    } else {
      // 如果为空，抛出错误
      ctx.status = 200;
      ctx.body = 'imageData不能为空';
    }
  } catch (error) {
    // 处理错误
    console.error(error);
    console.log(error)
    ctx.status = 500;
    ctx.body = error.message;
  }
});


// 加载模型
const load_model = async () => {
  // 三种加载模型的方法和配置
  // mobilenetv2: ['/quant_nsfw_mobilenet/'],
  // mobilenetMid: ['/quant_mid/', { type: 'graph' }],
  // inceptionv3: ['/model/', { size: 299 }],
  // 获取当前路径
  // const currentPath = path.resolve(__dirname,'..');
  _model = await nsfwjs.load(`http://localhost/models/quant_mid/model.json`,{ type: 'graph' })
}

load_model();

module.exports = router

