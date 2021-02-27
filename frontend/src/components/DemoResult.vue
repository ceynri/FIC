<template>
  <div class="demo_result">
    <div class="card">
      <div class="image_wrapper">
        <img class="image" :src="data.input" alt="input" ref="image" />
      </div>
      <div class="image_info">
        <div class="image_name">input image</div>
        <div class="image_data_info">{{ data.inputSize }}</div>
      </div>
    </div>
    <div class="card">
      <div class="image_wrapper">
        <img class="image" :src="data.output" alt="output" ref="image" />
      </div>
      <div class="image_info">
        <div class="image_name">output image</div>
        <div class="image_data_info">{{ data.compressedSize }} (Compressed data)</div>
        <div class="image_data_info">0.8 bpp</div>
      </div>
    </div>
    <div class="card">
      <div class="image_wrapper">
        <img class="image" :src="data.input" alt="jpeg compress" ref="image" />
      </div>
      <div class="image_info">
        <div class="image_name">JPEG image</div>
        <div class="image_data_info">{{ data.inputSize }}</div>
        <div class="image_data_info">0.8 bpp</div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  props: {
    data: {
      type: Object,
      required: true,
    },
  },
};
</script>

<style lang="scss" scoped>
.demo_result {
  padding: 40px 0;
  display: flex;
  justify-content: space-around;
  flex-wrap: wrap;

  .card {
    margin: 10px;
    .image_wrapper {
      .image {
        width: auto;
        height: auto;
        max-width: 100%;
        max-height: 100%;
      }
    }

    .image_info {
      color: var(--text2);
      margin: 16px 20px 20px;

      .image_name {
        @include no-wrap;
        color: var(--text);
        font-size: 16px;
        margin-bottom: 10px;
      }

      .image_data_info {
        font-size: 12px;
        margin-top: 8px;
      }
    }
  }
}
</style>
