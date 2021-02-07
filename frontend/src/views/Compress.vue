<template>
  <section class="compressor">
    <header class="title_wrapper">
      <h1 class="title">Compressor</h1>
      <div class="comment">Upload facial images to compress</div>
    </header>
    <Uploader accept="image/*" />
  </section>
</template>

<script>
import Uploader from '@/components/Uploader.vue';

export default {
  components: { Uploader },
};
</script>

<style lang="scss" scoped>
.compressor {
  margin: 0 auto;
  padding: 60px 0;
  width: 900px;

  .title {
    font-size: 32px;
    margin: 0;
    margin-bottom: 8px;
  }

  .comment {
    color: var(--text2);
    font-size: 20px;
    margin-bottom: 16px;
  }
}
</style>
