<template>
  <div class="demo_result">
    <div class="card">
      <div class="image_wrapper">
        <img class="image" :src="result.raw" alt="raw" ref="image" />
      </div>
      <div class="image_info">
        <div class="image_name">raw image</div>
        <div class="image_size">{{ result.rawResolution }}</div>
        <div class="image_size">{{ result.rawSize }}</div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  props: {
    result: {
      type: Object,
      required: true,
    },
  },
};
</script>

<style>
</style>