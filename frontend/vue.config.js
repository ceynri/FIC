module.exports = {
  pages: {
    index: {
      // page 入口
      entry: 'src/main.js',
      // 模板来源
      template: 'public/index.html',
      // 网页标题
      title: 'FIC - facial image compression tool base on deep learning',
    },
  },
  devServer: {
    // hot: true,
    port: 7211,
  },
  // 配置全局样式变量
  css: {
    loaderOptions: {
      sass: {
        prependData: '@import "@/styles/_mixin.scss";',
      },
    },
  },
};
